import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import dec_pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os
from datetime import timedelta

from utils import load_yaml_config, generate_training_data
from utils.graph import DecentralizedNetwork
from utils.results import plot_result

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
  

# distributed enviorment setup
def init_distributed_mode(backend='nccl', master_addr='localhost', master_port='12345'):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    dist.init_process_group(backend=backend, init_method='tcp://{}:{}'.format(master_addr, master_port), 
                            world_size=world_size, rank=rank, timeout=timedelta(seconds=1800))
    
    return world_size, rank


def create_augmented_dataset(local_x, local_y, world_size: int=1, rank: int=0, dataset_size: int=50, 
                                 partition_criteria: str='random'):
    """
    Create augmented dataset (D_c+) = local dataset (D_i) + global communication dataset (D_c)
    Args:
        local_x: Local training input data. (D_i)
        local_y: Local training output data. (D_i)
        world_size: Number of processes.
        rank: Current process rank.
        dataset_size: Size of the communication dataset to create.
        partition_criteria: Criteria for partitioning the dataset (default: 'random').
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
    torch.manual_seed(rank + 42)  # Ensure randomness acreoss different ranks

    dataset_size = int(local_x.size(0) / world_size)
    # dataset_size = min(dataset_size, local_x.size(0))  
    sample_indices = torch.randperm(local_x.size(0))[:dataset_size]
    local_comm_x = local_x[sample_indices]
    local_comm_y = local_y[sample_indices]

    # Step 2: communicate local communication dataset to other agents 
    # (flooding, BFS spanning tree, gossip, etc.)



    # create augmented dataset
    aug_x = torch.cat([local_x, comm_x], dim=0)
    aug_y = torch.cat([local_y, comm_y], dim=0)
    
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
    aug_x, aug_y = create_augmented_dataset(train_x, train_y, world_size, rank, dataset_size=50)
    
    # Train on augmented dataset with warm start
    likelihood = gpytorch.likelihoods.GaussianLikelihood() 
    model = ExactGPModel(aug_x, aug_y, likelihood)

    # move data to device
    model = model.to(device) 
    likelihood = likelihood.to(device)
    aug_x = aug_x.to(device)
    aug_y = aug_y.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = dec_pxadmm(model.parameters(), neighbors=neighbors, rho=admm_params['rho'],
                            lip=admm_params['lip'], rank=rank, world_size=world_size, 
                            tol_abs=admm_params['tol_abs'], tol_rel=admm_params['tol_rel'])
    
    def closure():
        optimizer.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005):
            output = model(aug_x)
            loss = -mll(output, aug_y)
            loss.backward()
            grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        return loss, grad
    
    model.train()
    likelihood.train()

    for epoch in range(admm_params['num_epochs']):
        optimizer.step(closure)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{admm_params['num_epochs']}, Loss: {closure()[0].item()}")
        # convergence check
    
    dist.destroy_process_group()
    return model, likelihood

def test_model(model, likelihood, test_x, device):
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

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = test_x.to(device)
        preds = model(test_x)
        mean = preds.mean
        lower, upper = preds.confidence_region()
    
    return mean.cpu(), lower.cpu(), upper.cpu()


if __name__ == "__main__":
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config_path = 'config/dec_gapxGP.yaml'
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))

    admm_params = {}
    admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lip'] = float(config.get('lip', 1.0))
    admm_params['tol_abs'] = float(config.get('tol_abs', 1e-6))
    admm_params['tol_rel'] = float(config.get('tol_rel', 1e-4))

    backend = str(config.get('backend', 'nccl'))
    graph_viz = bool(config.get('graph_viz', False))
    aug_dataset_size = int(config.get('aug_dataset_size', 50))

    # Generate training data
    local_x, local_y = generate_training_data(num_samples=num_samples, input_dim=input_dim, rank=rank,  
                                              world_size=world_size, partition='random')
        
    # get information about neighbors
    dec_graph = DecentralizedNetwork(num_nodes=world_size, graph_type='degree', dim=input_dim, degree=2, seed=42)
    neighbors = dec_graph.neighbors[rank]
    print(f"Rank {rank} neighbors: {neighbors}")

    if graph_viz and rank == 0:
        dec_graph.visualize_graph()

    # Train the model
    model, likelihood = train_model(local_x, local_y, device, admm_params, neighbors, backend=backend)

    # Test the model
    test_x = torch.linspace(0, 1, 100)
    mean, lower, upper = test_model(model, likelihood, test_x, device)

    # Plot the results
    plot_result(local_x, local_y, test_x, mean, lower, upper, rank=rank)