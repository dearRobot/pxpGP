import torch
import gpytorch
from admm import pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

from utils import generate_training_data, load_yaml_config
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
    
    # Step 2: communicate local communication dataset to central node rank 0
    sample_x_list = [torch.empty_like(local_comm_x) for _ in range(world_size)]
    sample_y_list = [torch.empty_like(local_comm_y) for _ in range(world_size)]

    dist.gather(local_comm_x, gather_list=sample_x_list if rank == 0 else None, dst=0)
    dist.gather(local_comm_y, gather_list=sample_y_list if rank == 0 else None, dst=0)

    # form communication dataset at rank 0 central node
    if rank == 0:
        comm_x = torch.cat(sample_x_list, dim=0)
        comm_y = torch.cat(sample_y_list, dim=0)
    else:
        comm_x = torch.zeros(dataset_size * world_size, dtype=local_x.dtype, device=local_x.device)
        comm_y = torch.zeros(dataset_size * world_size, dtype=local_y.dtype, device=local_y.device)


    # broadcast the communication dataset to all agents from rank 0
    dist.broadcast(comm_x, src=0)
    dist.broadcast(comm_y, src=0)

    # create augmented dataset
    aug_x = torch.cat([local_x, comm_x], dim=0)
    aug_y = torch.cat([local_y, comm_y], dim=0)

    # print(f"Rank {rank} - Augmented dataset size: {aug_x.size(0)}")
    
    return aug_x, aug_y


def train_model(model, likelihood, train_x, train_y, device, num_epochs: int=100, rho: float=0.8, 
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

    # Stage 1: Train local model on local dataset without consensus    
    world_size, rank = init_distributed_mode(backend=backend)

    # move data to device
    # model = model.to(device) 
    # likelihood = likelihood.to(device)
    # train_x = train_x.to(device)
    # train_y = train_y.to(device)

    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # optimizer = pxadmm(model.parameters(), rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel,
    #                    rank=rank, world_size=world_size)

    # def closure():
    #     optimizer.zero_grad()
    #     with gpytorch.settings.min_preconditioning_size(0.005):
    #         output = model(train_x)
    #         loss = -mll(output, train_y)
    #         loss.backward() 
    #         grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
    #     return loss, grad 
    
    # model.train()
    # likelihood.train()

    # for epoch in range(num_epochs):
    #     converged_ = optimizer.step(closure, consensus=False)
    #     if rank == 0 and (epoch + 1) % 10 == 0:
    #         print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {closure()[0].item()}") 
        
    #     if converged_:
    #         if rank == 0:
    #             print("Converged at epoch {}".format(epoch + 1))
    #         break

    # # Clear gradients and optimizer state
    # optimizer.zero_grad(set_to_none=True)
    # torch.cuda.empty_cache()
    
    # generate augmented dataset
    aug_x, aug_y = create_augmented_dataset(train_x, train_y, world_size, rank, dataset_size=50)
   
    # Stage 2: Train on augmented dataset with warm start
    likelihood_aug = gpytorch.likelihoods.GaussianLikelihood() 
    model_aug = ExactGPModel(aug_x, aug_y, likelihood_aug)

    # warm start
    # model_aug.mean_module.constant.data = model.mean_module.constant.data.clone()
    # model_aug.covar_module.base_kernel.lengthscale.data = model.covar_module.base_kernel.lengthscale.data.clone()
    # model_aug.covar_module.outputscale.data = model.covar_module.outputscale.data.clone()
    # likelihood_aug.noise.data = likelihood.noise.data.clone()     

    model_aug = model_aug.to(device)
    likelihood_aug = likelihood_aug.to(device)
    aug_x = aug_x.to(device)
    aug_y = aug_y.to(device)

    mll_aug = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_aug, model_aug)
    optimizer_aug = pxadmm(model_aug.parameters(), rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel,
                           rank=rank, world_size=world_size)
    
    def closure_aug():
        optimizer_aug.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005):
            output_aug = model_aug(aug_x)
            loss_aug = -mll_aug(output_aug, aug_y)
            loss_aug.backward() 
            grad_aug = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() 
                                  for p in model_aug.parameters()])
        return loss_aug, grad_aug
    
    model_aug.train()
    likelihood_aug.train()

    for epoch in range(num_epochs):
        converged_aug = optimizer_aug.step(closure_aug, consensus=True)
        
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {closure_aug()[0].item()}") 
        
        if converged_aug:
            if rank == 0:
                print("Converged at epoch {}".format(epoch + 1))
            break
    
    torch.cuda.empty_cache() 
    
    dist.destroy_process_group()
    return model_aug, likelihood_aug, aug_x, aug_y


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



if __name__ == "__main__":
      
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load yaml configuration
    config_path = 'config/gapxGP.yaml'
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    num_epochs = int(config.get('num_epochs', 100))
    rho = float(config.get('rho', 1.0))
    lip = float(config.get('lip', 1.0))
    tol_abs = float(config.get('tol_abs', 1e-6))
    tol_rel = float(config.get('tol_rel', 1e-4))
    backend = str(config.get('backend', 'nccl'))

    # generate local training data
    local_x, local_y = generate_training_data(num_samples=num_samples, input_dim=input_dim, rank=rank, 
                                                world_size=world_size, partition='sequential')

    # create the local model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()  
    model = ExactGPModel(local_x, local_y, likelihood)

    # OS enviorment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # train the model
    model, likelihood, aug_x, aug_y = train_model(model, likelihood, local_x, local_y, device, 
                                    num_epochs=num_epochs, rho=rho, lip=lip, tol_abs=tol_abs, 
                                    tol_rel=tol_rel, backend=backend)
    
    # test the model
    test_x = torch.linspace(0, 1, 1000)
    mean, lower, upper = test_model(model, likelihood, test_x, device)

    # plot the results
    # plot_result(local_x, local_y, test_x, mean, lower, upper, rank=rank)
    plot_result(aug_x, aug_y, test_x, mean, lower, upper, rank=rank)