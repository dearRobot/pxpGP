import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

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
   

# distributed enviorment
def init_distributed_mode(backend='nccl', master_addr='localhost', master_port='12345'):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    dist.init_process_group(backend=backend, init_method='tcp://{}:{}'.format(master_addr, master_port), 
                            world_size=world_size, rank=rank)

    return world_size, rank


def normalize_data(train_x, train_y):
    """
    Normalize the data to the range [0, 1]
    Args:
        data: Input data.
    Returns:
        Normalized data.
    """
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data - min_val) / (max_val - min_val)


def create_communication_dataset(train_x, train_y, world_size, rank):
    """
    Create communication dataset for distributed training
    Args:
        train_x: Local training input data.
        train_y: Local training output data.
        world_size: Number of processes.
        rank: Current process rank.
    Returns:
        aug_x : Local communication training input data.
        aug_y : Local communication training output data.
    """


def train_model(model, likelihood, train_x, train_y, num_epochs: int=100, rho: float=0.8, 
                            lip: float=1.0, tol_abs: float=1e-6, tol_rel: float=1e-4, backend='nccl'):
    """
    Train the model using pxADMM optimizer
    Args:
        model: The GP model to train.
        likelihood: The likelihood function.
        train_x: Training input data.
        train_y: Training output data.
        optimizer: The pxADMM optimizer.
        num_epochs: Number of training epochs.

    1. Each agent will train its local model with local dataset.(with or without z consensus update)
    2. Each agent will create its local sample dataset
    3. Share the local sample dataset with other agents as communication dataset
    4. Each agent will crate augmented dataset using local dataset + communication dataset
    5. Each agent will train its local model with augmented dataset again. (with or without z consensus update)
    """

# Train the local model with local dataset without z consensus update
    
    # intialize distributed training
    world_size, rank = init_distributed_mode(backend=backend)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    likelihood = likelihood.to(device)
    mll = mll.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    print("Rank {}: model parameters are {}".format(rank, [p.shape for p in model.parameters()]))

    # optimizer
    optimizer = pxadmm(model.parameters(), rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel,
                       rank=rank, world_size=world_size)

    def closure():
        optimizer.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005):
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward() 
            grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        return loss, grad 
    
    model.train()
    likelihood.train()

    for epoch in range(num_epochs):
        converged_ = optimizer.step(closure, consensus=False)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {closure()[0].item()}") 
        
        if converged_:
            if rank == 0:
                print("Converged at epoch {}".format(epoch + 1))
            break

    dist.destroy_process_group()
    return model, likelihood

def test_model(model, likelihood, test_x):
    """
    Test the model using pxADMM optimizer
    Args:
        model: The GP model to test.
        likelihood: The likelihood function.
        test_x: Testing input data.
    """
    model.eval()
    likelihood.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 1D Data
    train_x = torch.linspace(0, 1, 1000)
    train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2

    if rank == 0:
        print("Starting training...")
        print("global train_x shape:", train_x.shape)
        print("global train_y shape:", train_y.shape)

    # divide dataset into m parts based on world size and rank
    torch.manual_seed(42) 
    local_indices = torch.randperm(train_x.size(0))
    split_indices = torch.chunk(local_indices, world_size)
    local_indices = split_indices[rank]
    local_x = train_x[local_indices]
    local_y = train_y[local_indices]

    if rank == 0:
        print("local_x shape:", local_x.shape)
        print("local_y shape:", local_y.shape)

    # create the local model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()  
    model = ExactGPModel(local_x, local_y, likelihood)

    # OS enviorment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

     # train the model
    model, likelihood = train_model(model, likelihood, local_x, local_y, num_epochs=100,
                                    rho=0.8, lip=1.0, tol_abs=1e-5, tol_rel=1e-3, backend='gloo')
    
    # test the model
    test_x = torch.linspace(0, 1, 1000)
    mean, lower, upper = test_model(model, likelihood, test_x)

    # plot the results
    plot_results(local_x, local_y, test_x, mean, lower, upper)