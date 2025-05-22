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