import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import dec_admm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os
from datetime import timedelta
import numpy as np
import time
from filelock import FileLock
import json
from sklearn.model_selection import train_test_split

from utils import load_yaml_config, split_agent_data
from utils.graph import DecentralizedNetwork

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
    

# distributed enviorment setup
def init_distributed_mode(backend='nccl', master_addr='localhost', master_port='12345'):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    dist.init_process_group(backend=backend, init_method='tcp://{}:{}'.format(master_addr, master_port), 
                            world_size=world_size, rank=rank, timeout=timedelta(seconds=1800))
    
    return world_size, rank


def train_model(model, likelihood, train_x, train_y, device, admm_params, neighbors, backend='nccl'):
    """
    Train the Gaussian Process model using decentralized ADMM optimization.
    Args:
        model: The Gaussian Process model to train.
        likelihood: The likelihood function for the model.
        train_x: Training input data.
        train_y: Training output data.
        device: Device to run the training on (CPU or GPU).
        admm_params: Dictionary containing ADMM parameters like num_epochs, rho, etc.
        backend: Distributed backend to use (default: 'nccl').
    Returns:
        Trained model and likelihood.
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
    optimizer = dec_admm(model.parameters(), neighbors=neighbors, rho=admm_params['rho'], 
                            max_iter=admm_params['max_iter'], lr=admm_params['lr'], 
                            rank=rank, world_size=world_size)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        return loss, output  
    
    model.train()
    likelihood.train()
    
    for epoch in range(admm_params['num_epochs']):
        optimizer.step(closure)
        
        if not torch.isfinite(torch.tensor(closure()[0].item())):
            if rank == 0:
                print(f"Epoch {epoch + 1}: Loss is NaN, stopping early.")
            break
        
        if rank == 0 and epoch== 1:
            print(f"Rank {rank} - Epoch {epoch+1}/{admm_params['num_epochs']} loss: {closure()[0].item()}")
        
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Rank {rank} - Epoch {epoch+1}/{admm_params['num_epochs']} loss: {closure()[0].item()}")
        
    if rank == 0:
        print("Training complete.")
    
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
    rmse_error = torch.sqrt(torch.mean((mean - test_y) ** 2)).item()
    
    return mean.cpu(), lower.cpu(), upper.cpu(), rmse_error


if __name__ == "__main__":    
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    if world_size >= 36:
        device = 'cpu'
    else:    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load yaml configuration
    config_path = 'config/dec_cGP.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    dataset = int(config.get('dataset', 1))
    test_split = float(config.get('test_split', 0.2))

    admm_params = {}
    # admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lr'] = float(config.get('lr', 0.01))
    admm_params['max_iter'] = int(config.get('max_iter', 10))

    admm_params['num_epochs'] = int(min(world_size*2.0, 200))

    backend = str(config.get('backend', 'nccl'))
    graph_viz = bool(config.get('graph_viz', False))

    # load dataset
    datax_path = f'dataset/dataset{dataset}/dataset{dataset}x_{input_dim}d_{num_samples}.csv'
    datay_path = f'dataset/dataset{dataset}/dataset{dataset}y_{input_dim}d_{num_samples}.csv'

    if not os.path.exists(datax_path) or not os.path.exists(datay_path):
        raise FileNotFoundError(f"Dataset files {datax_path} or {datay_path} do not exist.")
    
    x = torch.tensor(np.loadtxt(datax_path, delimiter=',', dtype=np.float32))
    y = torch.tensor(np.loadtxt(datay_path, delimiter=',', dtype=np.float32))
    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_split, random_state=42)

    local_x, local_y = split_agent_data(x, y, world_size, rank, partition='sequential', input_dim=input_dim)
    
    # Create the local model and likelihood    
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) 
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(local_x, local_y, likelihood, kernel)

    # get information about neighbors
    dec_graph = DecentralizedNetwork(num_nodes=world_size, graph_type='degree', dim=input_dim, degree=2, seed=42)
    neighbors = dec_graph.neighbors[rank]
    # print(f"Rank {rank} neighbors: {neighbors}")

    if graph_viz and rank == 0:
        dec_graph.visualize_graph()

    # train the model 
    start_time = time.time()
    model, likelihood = train_model(model, likelihood, local_x, local_y, device, admm_params=admm_params,
                                    neighbors=neighbors, backend=backend)
    train_time = time.time() - start_time

    # test the model
    mean, lower, upper, rmse_error = test_model(model, likelihood, test_x, test_y, device)

    # print model and likelihood parameters
    if rank == 0:
        print(f"\033[92mRank {rank} - Testing RMSE: {rmse_error:.4f}\033[0m")
        if model.covar_module.base_kernel.lengthscale.numel() > 1:
            print(f"\033[92mRank: {rank}, Lengthscale:", model.covar_module.base_kernel.lengthscale.cpu().detach().numpy(), "\033[0m")  # Print all lengthscale values
        else:
            print(f"\033[92mRank: {rank}, Lengthscale:", model.covar_module.base_kernel.lengthscale.item(), "\033[0m")  # Print single lengthscale value
        
        print(f"\033[92mRank: {rank}, Outputscale:", model.covar_module.outputscale.item(), "\033[0m")
        print(f"\033[92mRank: {rank}, Noise:", model.likelihood.noise.item(), "\033[0m")

    result={
        'model': 'dec_cGP',
        'rank': rank,
        'world_size': world_size,
        'total_dataset_size': x.shape[0],
        'local_dataset_size': local_x.shape[0],
        'input_dim': input_dim,
        'lengthscale': model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().tolist(),
        'outputscale': model.covar_module.outputscale.item(),
        'noise': model.likelihood.noise.item(),
        'test_rmse': rmse_error,
        'train_time': train_time,
        'dataset': dataset
    }

    file_path = f'results/dataset_{dataset}/decentralized/dec_result_dim{input_dim}_agents_{world_size}_datasize_{x.shape[0]}.json'
    lock_path = file_path + '.lock'

    with FileLock(lock_path):
        with open(file_path, 'a') as f:
            f.write(json.dumps(result) + '\n')




