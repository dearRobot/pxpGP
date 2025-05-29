import torch
import gpytorch
from admm import pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

from utils import generate_training_data, load_yaml_config
from utils.results import plot_pxpGP_result, plot_result

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


# sparse GP Model with inducing or pseudo points/variational distribution
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, 
                                    variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# distributed enviorment
def init_distributed_mode(backend='nccl', master_addr='localhost', master_port='12345'):
    # Intialize distributed training
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    dist.init_process_group(backend=backend, init_method='tcp://{}:{}'.format(master_addr, master_port), 
                            world_size=world_size, rank=rank)

    return world_size, rank


def create_local_pseudo_dataset(local_x, local_y, device, dataset_size: int=50, rank: int=0, num_epochs: int=100):
    """
    Create local pseudo dataset (D_i) = local dataset (D_i)
    Args:
        local_x: Local training input data. (D_i)
        local_y: Local training output data. (D_i)
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        dataset_size: Size of the pseudo dataset to create.
        rank: Current process rank.
        num_epochs: Number of training epochs for the local sparse GP model.
    Returns:
        local_pseudo_x : Local pseudo training input data. (D_i)
        local_pseudo_y : Local pseudo training output data. (D_i)
    """
    torch.manual_seed(rank + 42)  
    
    dataset_size = int(local_x.size(0) / world_size)
    # dataset_size = min(dataset_size, int(local_x.size(0)/world_size))  
    sample_indices = torch.randperm(local_x.size(0))[:dataset_size]
    
    local_pseudo_x = local_x[sample_indices]
    
    model_sparse = SparseGPModel(local_pseudo_x)
    likelihood_sparse = gpytorch.likelihoods.GaussianLikelihood() 

    # move data to device
    model_sparse = model_sparse.to(device)
    likelihood_sparse = likelihood_sparse.to(device)
    local_pseudo_x = local_pseudo_x.to(device)

    mll_sparse = gpytorch.mlls.VariationalELBO(likelihood_sparse, model_sparse, num_data=local_x.size(0))
        
    optimizer_sparse = torch.optim.Adam(model_sparse.parameters(), lr=0.03)

    # optimizer_sparse = pxadmm(model_sparse.parameters(), rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel,
    #                            rank=rank, world_size=world_size)
    
    def closure_sparse():
        optimizer_sparse.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005):
            output_sparse = model_sparse(local_x)
            loss_sparse = -mll_sparse(output_sparse, local_y)
            loss_sparse.backward()
            grad_sparse = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                                    for p in model_sparse.parameters()])
        return loss_sparse, grad_sparse
    
    model_sparse.train()
    likelihood_sparse.train()

    for epoch in range(num_epochs):
        # converged_sparse = optimizer_sparse.step(closure_sparse, consensus=False)
        optimizer_sparse.step(closure_sparse)
        if rank == 0 and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {closure_sparse()[0].item()}") 
        
        # if converged_sparse:
        #     if rank == 0:
        #         print("Converged at epoch {}".format(epoch + 1))
        #     break

    local_pseudo_x = model_sparse.variational_strategy.inducing_points.detach().clone()
    local_pseudo_x = local_pseudo_x.squeeze(-1) 

    # local_pseudo_x = torch.clamp(local_pseudo_x, min=local_x.min(), max=local_x.max())

    # clear gradients
    optimizer_sparse.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    # compute local_pseudo_y using the local sparse GP model
    model_sparse.eval()
    likelihood_sparse.eval()

    local_pseudo_y, _, _ = test_model(model_sparse, likelihood_sparse, local_pseudo_x, device)    
   
    
    print(f"Rank {rank} - Local pseudo dataset size: {local_pseudo_x.size(0)}")
    
    
    return local_pseudo_x, local_pseudo_y, model_sparse, likelihood_sparse



def create_augmented_dataset(local_x, local_y, device, world_size: int=1, rank: int=0, dataset_size: int=50, 
                                num_epochs: int = 100, partition_criteria: str='random'):
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
    
    # Step 1: create local pseudo dataset    
    local_x = local_x.to(device)
    local_y = local_y.to(device)
    
    local_pseudo_x, local_pseudo_y, model_sparse, _ = create_local_pseudo_dataset(local_x, local_y,
                            device, dataset_size=dataset_size, rank=rank, num_epochs=num_epochs)
    
    
    # Step 2: gather local pseudo dataset from all processes and create global pseudo dataset
    sample_x_list = [torch.empty_like(local_pseudo_x) for _ in range(world_size)]
    sample_y_list = [torch.empty_like(local_pseudo_y) for _ in range(world_size)]

    dist.gather(local_pseudo_x, gather_list=sample_x_list if rank == 0 else None, dst=0)
    dist.gather(local_pseudo_y, gather_list=sample_y_list if rank == 0 else None, dst=0)

    if rank == 0:
        comm_x = torch.cat(sample_x_list, dim=0)
        comm_y = torch.cat(sample_y_list, dim=0)
    else:
        comm_x = torch.zeros(dataset_size * world_size, dtype=local_x.dtype, device=device)
        comm_y = torch.zeros(dataset_size * world_size, dtype=local_y.dtype, device=device)

    # broadcast the communication dataset to all agents from rank 0
    dist.broadcast(comm_x, src=0)
    dist.broadcast(comm_y, src=0)
    
    # create augmented dataset
    pseudo_x = torch.cat([local_x, comm_x], dim=0)
    pseudo_y = torch.cat([local_y, comm_y], dim=0)

    # Step 3: Share the local model hyperparameters with the central node (rank 0) to form average
    local_hyperparams = torch.tensor([model_sparse.mean_module.constant.item(),
                                    model_sparse.covar_module.base_kernel.lengthscale.item(),
                                    model_sparse.covar_module.outputscale.item()], dtype=torch.float32, device=device)

    hyperparams_list = [torch.empty_like(local_hyperparams) for _ in range(world_size)] 
    dist.gather(local_hyperparams, gather_list=hyperparams_list if rank == 0 else None, dst=0)

    if rank == 0:   
        hyperparam_stack = torch.stack(hyperparams_list)
        avg_hyperparams_ = hyperparam_stack.mean(dim=0)
    else:
        avg_hyperparams_ = torch.zeros_like(local_hyperparams, dtype=torch.float32, device=device)

    dist.broadcast(avg_hyperparams_, src=0)

    avg_hyperparams = {'mean_constant': avg_hyperparams_[0].item(),
                        'lengthscale': avg_hyperparams_[1].item(),
                        'outputscale': avg_hyperparams_[2].item()}
    
    local_hyperpar = {'mean_constant': model_sparse.mean_module.constant.item(),
                        'lengthscale': model_sparse.covar_module.base_kernel.lengthscale.item(),
                        'outputscale': model_sparse.covar_module.outputscale.item()}
    
    torch.cuda.empty_cache()
    return pseudo_x, pseudo_y, local_hyperpar #avg_hyperparams



def train_model(model, likelihood, train_x, train_y, device, num_epochs: int=100, rho: float=0.8, 
                        lip: float=1.0, tol_abs: float=1e-6, tol_rel: float=1e-4, backend='nccl'):
    """
    Train the model using pxADMM optimizer
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

    1. Each agent will train its local sparse model with local dataset and find optimal inducing points.
    2. Each agent will share its local inducing points with central node (rank 0).
    3. Central node will create a global inducing points dataset by concatenating all local inducing points.
    4. Each agent will receive the global inducing points dataset from central node.
    5. Each agent will create augmented dataset using local dataset + global inducing points dataset.
            OR just use the global inducing points dataset as augmented dataset.
    6. Each agent will train its local model with augmented dataset again.
    """

    # Stage 1: Train local sparse GP model on local dataset and find optimal inducing points   
    world_size, rank = init_distributed_mode(backend=backend)

    pseudo_x, pseudo_y, avg_hyperparams = create_augmented_dataset(train_x, train_y, device, world_size=world_size,
                            rank=rank, dataset_size=50, num_epochs=200, partition_criteria='random')

    
    print(f"Rank {rank} - Pseudo dataset size: {pseudo_x.size(0)}")
    
    # Stage 2: Train on augmented dataset with warm start
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(pseudo_x, pseudo_y, likelihood)

    # warm start
    model.mean_module.constant.data = avg_hyperparams['mean_constant'] * torch.ones_like(model.mean_module.constant.data)
    model.covar_module.base_kernel.lengthscale.data = avg_hyperparams['lengthscale'] * torch.ones_like(model.covar_module.base_kernel.lengthscale.data)
    model.covar_module.outputscale.data = avg_hyperparams['outputscale'] * torch.ones_like(model.covar_module.outputscale.data)
    
    model = model.to(device)
    likelihood = likelihood.to(device)
    pseudo_x = pseudo_x.to(device)
    pseudo_y = pseudo_y.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = pxadmm(model.parameters(), rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel,
                            rank=rank, world_size=world_size)
    
    def closure():
        optimizer.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005):
            output = model(pseudo_x)
            loss = -mll(output, pseudo_y)
            loss.backward()
            grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                              for p in model.parameters()])
        return loss, grad
    
    model.train()
    likelihood.train()

    for epoch in range(num_epochs):
        converged = optimizer.step(closure, consensus=True)
        
        if rank == 0 and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {closure()[0].item()}")
        if converged:
            if rank == 0:
                print("Converged at epoch {}".format(epoch + 1))
            break

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()    

    dist.destroy_process_group()
    return model, likelihood, pseudo_x, pseudo_y


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
    config_path = 'config/pxpGP.yaml'
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
                                                world_size=world_size, partition='random')

    # create the local model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()  
    model = ExactGPModel(local_x, local_y, likelihood)

    # OS enviorment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # train the model
    model, likelihood, pseudo_x, pseudo_y = train_model(model, likelihood, local_x, local_y, device, 
                                        num_epochs=num_epochs, rho=rho, lip=lip, tol_abs=tol_abs, 
                                        tol_rel=tol_rel, backend=backend)
    
    
    # test the model
    test_x = torch.linspace(0, 1, 1000)
    mean, lower, upper = test_model(model, likelihood, test_x, device)

    # plot the results
    # plot_result(local_x, local_y, test_x, mean, lower, upper, rank=rank)
    plot_result(pseudo_x, pseudo_y, test_x, mean, lower, upper, rank=rank)

