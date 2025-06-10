import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import dec_admm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

from utils import load_yaml_config, generate_training_data
from utils.graph import DecentralizedNetwork
from datetime import timedelta

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

    if rank == 0:
        print(f"Starting training with {world_size} processes, using backend: {backend}")
    
    for epoch in range(admm_params['num_epochs']):
        optimizer.step(closure)
        
        if rank == 0:
            print(f"Rank {rank} - Epoch {epoch+1}/{admm_params['num_epochs']} loss: {closure()[0].item()}")
        
        # if rank == 0 and (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch+1}/{admm_params['num_epochs']} Loss: {closure()[0].item()}")  
      
    
    if rank == 0:
        print("Training complete.")
    
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache() 

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
        Predictions and confidence intervals.
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = test_x.to(device)
        preds = model(test_x)
        mean = preds.mean
        lower, upper = preds.confidence_region()

    return mean, lower, upper


if __name__ == "__main__":    
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load yaml configuration
    config_path = 'config/dec_cGP.yaml'
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))

    admm_params = {}
    admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lr'] = float(config.get('lr', 0.01))
    admm_params['max_iter'] = int(config.get('max_iter', 10))

    backend = str(config.get('backend', 'nccl'))
    graph_viz = bool(config.get('graph_viz', False))

    # # OS enviorment variables for distributed training
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'

    # generate local training data
    local_x, local_y = generate_training_data(num_samples=num_samples, input_dim=input_dim, rank=rank, 
                                                world_size=world_size, partition='sequential')
    
    # Create the local model and likelihood    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(local_x, local_y, likelihood)

    # get information about neighbors
    dec_graph = DecentralizedNetwork(num_nodes=world_size, graph_type='degree', dim=input_dim, degree=2, seed=42)
    neighbors = dec_graph.neighbors[rank]
    print(f"Rank {rank} neighbors: {neighbors}")

    if graph_viz and rank == 0:
        dec_graph.visualize_graph()

    
    model, likelihood = train_model(model, likelihood, local_x, local_y, device, admm_params=admm_params,
                                    neighbors=neighbors, backend=backend)


