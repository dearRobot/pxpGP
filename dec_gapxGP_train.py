import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import dec_pxadmm
import torch.distributed as dist
from linear_operator.settings import max_cg_iterations, cg_tolerance, cholesky_jitter
import os
from datetime import timedelta
from sklearn.model_selection import train_test_split
import time
import json
import numpy as np
from filelock import FileLock


from utils import load_yaml_config, generate_dataset, split_agent_data
from utils.graph import DecentralizedNetwork
from utils.results import plot_result
torch.cuda.empty_cache()

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


def broadcast_data(data, neighbors, rank: int=0, world_size: int=1):
    """
    Broadcast data to all neighbors in the network.
    Args:
        data: Data to be broadcasted.
        neighbors: List of neighboring agents.
        rank: Current process rank.
        world_size: Number of processes.
    Returns:
        broadcasted_data: Broadcasted data.
    """
    if rank == 0:
        for neighbor in neighbors:
            dist.send(data, dst=neighbor)
    else:
        dist.recv(data, src=0)
    
    return data
    
    
def create_augmented_dataset(local_x, local_y, world_size: int=1, rank: int=0, dataset_size: int=50, 
                                neighbors: list=None):
    """
    Create augmented dataset (D_c+) = local dataset (D_i) + global communication dataset (D_c)
    Args:
        local_x: Local training input data. (D_i)
        local_y: Local training output data. (D_i)
        world_size: Number of processes.
        rank: Current process rank.
        dataset_size: Size of the communication dataset to create.
    Returns:
        aug_x : Augmented training input data. (D_c)
        aug_y : Augmented training output data. (D_c)
    """

    if not isinstance(rank, int) or rank < 0:
        raise ValueError("Rank must be a non-negative integer.")
    if not isinstance(dataset_size, int) or dataset_size <= 0:
        raise ValueError("Dataset size must be a positive integer.")
    if world_size <= 0:
        raise ValueError("World size must be greater than 0.")
    
    # Step 1: create local communication dataset
    random_int = torch.randint(0, 1000, (1,)).item() 
    torch.manual_seed(random_int + rank)  # Ensure randomness acreoss different ranks

    # make sure dataset size is same for all ranks
    if rank == 0:
        dataset_size = min(int(local_x.size(0) // world_size), int(local_x.size(0) // 10))
        dataset_size = max(dataset_size, 4)
    else:
        dataset_size = 0

    dataset_size_tensor = torch.tensor(dataset_size, device=device)
    dist.broadcast(dataset_size_tensor, src=0)
    dataset_size = dataset_size_tensor.item()

    sample_indices = torch.randperm(local_x.size(0))[:dataset_size]
    local_comm_x = local_x[sample_indices]
    local_comm_y = local_y[sample_indices]

    # Step 2: communicate local communication dataset to other agents 
# TODO: (flooding, BFS spanning tree, gossip, etc.)
    
    # broadcast_data(local_pseudo_x, neighbors, rank=rank, world_size=world_size)
    # broadcast_data(local_pseudo_y, neighbors, rank=rank, world_size=world_size)
    # broadcast_data(local_hyperparams, neighbors, rank=rank, world_size=world_size)
    
    sample_x_list = [torch.empty_like(local_comm_x) for _ in range(world_size)]
    sample_y_list = [torch.empty_like(local_comm_y) for _ in range(world_size)]

    dist.gather(local_comm_x, gather_list=sample_x_list if rank == 0 else None, dst=0)
    dist.gather(local_comm_y, gather_list=sample_y_list if rank == 0 else None, dst=0)

    # form communication dataset at rank 0 central node
    if rank == 0:
        comm_x = torch.cat(sample_x_list, dim=0)
        comm_y = torch.cat(sample_y_list, dim=0)
    else:
        comm_x = torch.zeros((dataset_size * world_size, *local_x.shape[1:]), dtype=local_x.dtype, device=local_x.device) 
        comm_y = torch.zeros(dataset_size * world_size, dtype=local_y.dtype, device=local_y.device)

    # broadcast the communication dataset to all agents from rank 0
    dist.broadcast(comm_x, src=0)
    dist.broadcast(comm_y, src=0)

    # create augmented dataset
    aug_x = torch.cat([local_x, comm_x], dim=0)
    aug_y = torch.cat([local_y, comm_y], dim=0)
    
    # remove duplicate entries in the augmented dataset
    aug_x_np = aug_x.cpu().numpy()           # shape (N, d)
    aug_y_np = aug_y.cpu().numpy()  

    unique_rows, unique_idx = np.unique(aug_x_np, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    aug_x = torch.from_numpy(aug_x_np[unique_idx]).to(local_x.dtype).to(local_x.device)
    aug_y = torch.from_numpy(aug_y_np[unique_idx]).to(local_y.dtype).to(local_y.device)
       
    return aug_x, aug_y


def train_model(train_x, train_y, device, admm_params, neighbors, backend='nccl'):
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

    1. Each agent will create its local sample dataset
    2. Share the local sample dataset with other agents as communication dataset
    3. Each agent will crate augmented dataset using local dataset + communication dataset
    4. Each agent will train its local model with augmented dataset again. (with z consensus update)
    """
    
    # Initialize distributed training
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12345')
    world_size, rank = init_distributed_mode(backend=backend, master_addr=master_addr, 
                                              master_port=master_port)
    
    # generate augmented dataset
    aug_x, aug_y = create_augmented_dataset(train_x, train_y, world_size, rank, dataset_size=50,
                                            neighbors=neighbors)
    
    # Train on augmented dataset with warm start
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    likelihood_aug = gpytorch.likelihoods.GaussianLikelihood() 
    model_aug = ExactGPModel(aug_x, aug_y, likelihood_aug, kernel) 

    model_aug = model_aug.to(device)
    likelihood_aug = likelihood_aug.to(device)
    aug_x = aug_x.to(device)
    aug_y = aug_y.to(device)

    mll_aug = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_aug, model_aug)
    optimizer_aug = dec_pxadmm(model_aug.parameters(), neighbors=neighbors, rho=admm_params['rho'],
                            lip=admm_params['lip'], rank=rank, world_size=world_size, 
                            tol_abs=admm_params['tol_abs'], tol_rel=admm_params['tol_rel'])
    
    def closure_aug():
        optimizer_aug.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005), max_cg_iterations(2000), cg_tolerance(1e-2):
            output_aug = model_aug(aug_x)
            loss_aug = -mll_aug(output_aug, aug_y)
            loss_aug.backward() 
            grad_aug = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() 
                                  for p in model_aug.parameters()])
        return loss_aug, grad_aug
    
    model_aug.train()
    likelihood_aug.train()

    for epoch in range(admm_params['num_epochs']):
        converged = optimizer_aug.step(closure_aug, epoch=epoch)
        loss_val = closure_aug()[0].item()
        
        if not torch.isfinite(torch.tensor(loss_val)):
            if rank == 0:
                print(f"Epoch {epoch + 1}: Loss is NaN, stopping early.")
            break

        if converged:
            # if rank == 0:
            print(f"Rank {rank} - Converged at epoch {epoch + 1}, loss: {loss_val:.4f}")
            break

        if rank == 0 and epoch== 1:
            print(f"Rank {rank} - Epoch {epoch+1}/{admm_params['num_epochs']} loss: {closure_aug()[0].item()}")
        
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Rank {rank} - Epoch {epoch+1}/{admm_params['num_epochs']} loss: {closure_aug()[0].item()}")

    
    optimizer_aug.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()  
    
    dist.destroy_process_group()
    return model_aug, likelihood_aug


def test_model(model, likelihood, test_x, test_y, device):
    """
    Test the Gaussian Process model on test data.
    Args:
        model: The trained Gaussian Process model.
        likelihood: The likelihood function for the model.
        test_x: Test input data.
        device: Device to run the testing on (CPU or GPU).
    Returns:
        Predictions and uncertainties for the test data.
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
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    if world_size >= 36:
        device = 'cpu'
    else:    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config_path = 'config/dec_gapxGP.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    dataset = int(config.get('dataset', 1))
    test_split = float(config.get('test_split', 0.05))

    admm_params = {}
    # admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lip'] = float(config.get('lip', 1.0))
    admm_params['tol_abs'] = float(config.get('tol_abs', 1e-6))
    admm_params['tol_rel'] = float(config.get('tol_rel', 1e-4))

    admm_params['num_epochs'] = int(min(world_size*2.0, 500))

    backend = str(config.get('backend', 'nccl'))
    graph_viz = bool(config.get('graph_viz', False))
    aug_dataset_size = int(config.get('aug_dataset_size', 50))

    # load dataset
    datax_path = f'dataset/dataset{dataset}/dataset1x_{input_dim}d_{num_samples}.csv'
    datay_path = f'dataset/dataset{dataset}/dataset1y_{input_dim}d_{num_samples}.csv'

    if not os.path.exists(datax_path) or not os.path.exists(datay_path):
        raise FileNotFoundError(f"Dataset files {datax_path} or {datay_path} do not exist.")
    
    x = torch.tensor(np.loadtxt(datax_path, delimiter=',', dtype=np.float32))
    y = torch.tensor(np.loadtxt(datay_path, delimiter=',', dtype=np.float32))

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_split, random_state=42)
    local_x, local_y = split_agent_data(x, y, world_size, rank, input_dim=input_dim, partition='sequential')    
   
    # get information about neighbors
    dec_graph = DecentralizedNetwork(num_nodes=world_size, graph_type='degree', dim=input_dim, degree=2, seed=42)
    neighbors = dec_graph.neighbors[rank]
    # print(f"Rank {rank} neighbors: {neighbors}")

    if graph_viz and rank == 0:
        dec_graph.visualize_graph()

    # Train the model
    start_time = time.time()
    model, likelihood = train_model(local_x, local_y, device, admm_params, neighbors, backend=backend)
    train_time = time.time() - start_time

    # Test the model
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
        'model': 'dec_gapxGP',
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
