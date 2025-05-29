import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import cadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

from utils import load_yaml_config, generate_training_data

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# local GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# global GP model used for prediction as a combination of local models
class GlobalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, global_covar_inv):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.global_covar_inv = global_covar_inv
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# distributed enviorment
def init_distributed_mode(backend='nccl', master_addr='localhost', master_port='12345'):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    dist.init_process_group(backend=backend, init_method='tcp://{}:{}'.format(master_addr, master_port), 
                            world_size=world_size, rank=rank)

    return world_size, rank


def train_model(model, likelihood, train_x, train_y, device, num_epochs: int=100, rho: float=0.8, 
                lr: float=0.01, max_iter: int=100, backend='nccl'):
    """
    Train the Gaussian Process model using ADMM optimization.
    Args:
        model: The Gaussian Process model to train.
        likelihood: The likelihood function for the model.
        train_x: Training input data.
        train_y: Training output data.
        num_epochs: Number of training epochs.
        rho: ADMM parameter for convergence.
        lip: Lipschitz constant for the kernel function.
        tol_abs: Absolute tolerance for convergence.
        tol_rel: Relative tolerance for convergence.
        backend: Distributed backend to use ('nccl', 'gloo', etc.).
    Returns:
        model: The trained Gaussian Process model.
        likelihood: The likelihood function for the model.
    """
    
    # Initialize distributed training
    world_size, rank = init_distributed_mode(backend=backend)
    
    # move data to device
    model = model.to(device) 
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = cadmm(model.parameters(), lr=lr, max_iter=max_iter, rho=rho, rank=rank, world_size=world_size)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        return loss, output
    
    model.train()
    likelihood.train()

    for epoch in range(num_epochs):
        optimizer.step(closure)

        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} Loss: {closure()[0].item()}")  

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache() 

    dist.destroy_process_group()
    return model, likelihood


def test_model(model, likelihood, test_x, device):
    """
    Test the model using pxADMM optimizer
    Args:
        model: The GP model to test.
        likelihood: The likelihood function.
        test_x: Testing input data.
        device: Device to run the model on (CPU or GPU).
    Returns:
        mean: Predicted mean of the test data.
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = test_x.to(device)
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    return mean, lower, upper


def plot_results(train_x, train_y, test_x, mean, lower, upper):
    plt.figure(figsize=(12, 6))
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Train Data')
    plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b', label='Mean Prediction')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, color='blue', label='Confidence Interval')
    plt.title('Gaussian Process Regression Rank {}'.format(os.environ['RANK']))
    plt.legend()
    plt.show()


if __name__ == "__main__":
      
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load yaml configuration
    config_path = 'config/cGP.yaml'
    config = load_yaml_config(config_path)
    
    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    num_epochs = int(config.get('num_epochs', 100))
    rho = float(config.get('rho', 1.0))
    lr = float(config.get('lr', 0.001))
    max_iter = int(config.get('max_iter', 100))
    backend = str(config.get('backend', 'nccl'))
    
    # generate local training data
    local_x, local_y = generate_training_data(num_samples=num_samples, input_dim=input_dim, rank=rank, 
                                                world_size=world_size, partition='random')


    # Create the local model and likelihood    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(local_x, local_y, likelihood)

    # os environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Train the local model
    model, likelihood = train_model(model, likelihood, local_x, local_y, device, num_epochs=num_epochs, 
                                    rho=rho, lr=lr, max_iter=max_iter, backend=backend)
    
    
    # Test the model
    test_x = torch.linspace(0, 1, 51)
    mean, lower, upper = test_model(model, likelihood, test_x, device)

    # Plot the local results
    plot_results(local_x, local_y, test_x, mean, lower, upper)



