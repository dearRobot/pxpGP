import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import cadmm
import torch.distributed as dist
import os
from sklearn.model_selection import train_test_split

from utils import load_yaml_config, generate_training_data
from utils import generate_dataset, split_agent_data
from utils.results import plot_result, save_params



# local GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
    
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


def train_model(model, likelihood, train_x, train_y, device, admm_params, backend='nccl'):
    """
    Train the Gaussian Process model using ADMM optimization.
    Args:
        model: The Gaussian Process model to train.
        likelihood: The likelihood function for the model.
        train_x: Training input data.
        train_y: Training output data.
        device: Device to run the training on (CPU or GPU).
        admm_params: Dictionary containing ADMM parameters like num_epochs, rho, etc.
        backend: Distributed backend to use ('nccl', 'gloo', etc.).
    Returns:
        model: The trained Gaussian Process model.
        likelihood: The likelihood function for the model.
    """
    
    # Initialize distributed training
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12345')
    world_size, rank = init_distributed_mode(backend=backend, master_addr=master_addr, 
                                              master_port=master_port)
    
    # move data to device
    model = model.to(device) 
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = cadmm(model.parameters(), lr=admm_params['lr'], max_iter=admm_params['max_iter'],
                      rho=admm_params['rho'], rank=rank, world_size=world_size)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        return loss, output
    
    model.train()
    likelihood.train()

    for epoch in range(admm_params['num_epochs']):
        optimizer.step(closure)
        # optimizer.zero_grad()
        # output = model(train_x)
        # loss = -mll(output, train_y)
        # loss.backward()
        # optimizer.step()

        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{admm_params['num_epochs']} Loss: {closure()[0].item()}")  

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache() 

    dist.destroy_process_group()
    return model, likelihood


def test_model(model, likelihood, test_x, test_y, device):
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
    test_x = test_x.to(device)
    test_y = test_y.to(device) 

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    # compute RMSE error
    torch.sqrt(torch.mean((mean - test_y) ** 2)).item()
    print(f"RMSE: {torch.sqrt(torch.mean((mean - test_y) ** 2)).item():.4f}")
    
    return mean.cpu(), lower.cpu(), upper.cpu()


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
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    test_split = float(config.get('test_split', 0.2))

    admm_params = {}
    admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lr'] = float(config.get('lr', 0.01))
    admm_params['max_iter'] = int(config.get('max_iter', 100))

    backend = str(config.get('backend', 'nccl'))
    
    # generate local training data
    x, y = generate_dataset(num_samples, input_dim)    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_split, random_state=42)

    # split data among agents
    local_x, local_y = split_agent_data(x, y, world_size, rank, partition='sequential')

    # Create the local model and likelihood   
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) 
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(local_x, local_y, likelihood, kernel)

    # os environment variables for distributed training
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'

    # Train the local model
    model, likelihood = train_model(model, likelihood, local_x, local_y, device, admm_params, backend=backend)

    mean, lower, upper = test_model(model, likelihood, test_x, test_y, device)

    # print model and likelihood parameters
    if model.covar_module.base_kernel.lengthscale.numel() > 1:
        print(f"Rank: {rank}, Lengthscale:", model.covar_module.base_kernel.lengthscale.cpu().detach().numpy())  # Print all lengthscale values
    else:
        print(f"Rank: {rank}, Lengthscale:", model.covar_module.base_kernel.lengthscale.item())  # Print single lengthscale value
    
    print(f"Rank: {rank}, Outputscale:", model.covar_module.outputscale.item())
    print(f"Rank: {rank}, Noise:", model.likelihood.noise.item())
    
    # Save model and likelihood parameters  
    # torch.save(model.state_dict(), f'results/cGP_model_{input_dim}_rank_{rank}.pth')
    
    save_params(model, rank, input_dim, method='cGP',
                filepath=f'results/cGP_params_{input_dim}_rank_{rank}.json')



    # mean = mean.reshape(100, 100)
    # lower = lower.reshape(100, 100)
    # upper = upper.reshape(100, 100)

    # # Plot the results
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), mean.cpu().numpy(), cmap='viridis', edgecolor='k')
    # ax.set_xlabel('X1')
    # ax.set_ylabel('X2')
    # ax.set_zlabel('f(x1, x2)')
    # ax.set_title('GP Prediction with RBF Kernel (ARD)')
    # plt.tight_layout()
    # plt.show()


