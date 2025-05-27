import torch
import gpytorch
from admm import pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

from utils import generateTrainingData, loadYAMLConfig
from utils.results import plot_result

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


def create_communication_dataset(local_x, local_y, world_size: int=1, rank: int=0, dataset_size: int=50, 
                                 partition_criteria: str='random'):
    """
    Create communication dataset for distributed training
    Args:
        local_x: Local training input data.
        local_y: Local training output data.
        world_size: Number of processes.
        rank: Current process rank.
        dataset_size: Size of the communication dataset to create.
        partition_criteria: Criteria for partitioning the dataset (default: 'random').
    Returns:
        aug_x : Local communication training input data.
        aug_y : Local communication training output data.
    """
    
    print(f"Rank {rank} creating communication dataset...")
    
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("Rank must be a non-negative integer.")
    if not isinstance(dataset_size, int) or dataset_size <= 0:
        raise ValueError("Dataset size must be a positive integer.")
    if world_size <= 0:
        raise ValueError("World size must be greater than 0.")


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

    # generate local sample communication dataset
    comm_x, comm_y = create_communication_dataset(train_x, train_y, world_size, rank, dataset_size=50)
    
    
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



if __name__ == "__main__":
      
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # load yaml configuration
    config_path = 'config/gapxGP.yaml'
    config = loadYAMLConfig(config_path)

    num_epochs = int(config.get('num_epochs', 100))
    rho = float(config.get('rho', 1.0))
    lip = float(config.get('lip', 1.0))
    tol_abs = float(config.get('tol_abs', 1e-6))
    tol_rel = float(config.get('tol_rel', 1e-4))
    backend = str(config.get('backend', 'nccl'))

    # generate local training data
    local_x, local_y = generateTrainingData(num_samples=1000, input_dim=1, rank=rank, 
                                                world_size=world_size)

    # create the local model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()  
    model = ExactGPModel(local_x, local_y, likelihood)

    # OS enviorment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

     # train the model
    model, likelihood = train_model(model, likelihood, local_x, local_y, num_epochs=num_epochs,
                                    rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel, backend=backend)
    
    # test the model
    test_x = torch.linspace(0, 1, 1000)
    mean, lower, upper = test_model(model, likelihood, test_x)

    # plot the results
    plot_result(local_x, local_y, test_x, mean, lower, upper, rank=rank)