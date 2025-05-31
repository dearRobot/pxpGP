import torch
import gpytorch
from admm import pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

from utils import generate_training_data, generate_test_data, load_yaml_config
from utils.results import plot_pxpGP_result, plot_result

from sklearn.cluster import KMeans

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


def inducing_penalty(inducing_x, x_min: float=0.0, x_max: float=1.0, margin=0.01):
    below = torch.relu(x_min - inducing_x + margin)
    above = torch.relu(inducing_x - x_max + margin)
    return (below**2 + above**2).sum()



def create_local_pseudo_dataset(local_x, local_y, device, dataset_size: int=50, world_size: int=1, rank: int=0, num_epochs: int=100):
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
    
    # local_x = local_x.to(device)
    # local_y = local_y.to(device)
    
    # option to select initial dataset: random , k-means
    # local_pseudo_x = local_x[::10].unsqueeze(-1).to(device)

    if local_x.size(0) < dataset_size:
        raise ValueError(f"Local dataset size {local_x.size(0)} is smaller than dataset_size {dataset_size}.")
    
    x_min = local_x.min().item()
    x_max = local_x.max().item()
    
    # indices = torch.randperm(local_x.size(0))[:dataset_size]
    # local_pseudo_x = local_x[indices].unsqueeze(-1).to(device)  # unsqueeze to make it 2D

    # print(f"Rank {rank} - Local pseudo dataset is :{local_pseudo_x}")
    # print(f"Rank {rank} - Local pseudo dataset size: {local_pseudo_x.size(0)}")

    # normalize local_x before clustering and then denormalize the pseudo dataset
    # x_mean, x_std = local_x.mean(), local_x.std()
    # local_x_normalized = (local_x - x_mean) / (x_std + 1e-6)
    # local_y_normalized = (local_y - local_y.mean()) / (local_y.std() + 1e-6)


    # check how noramalized the local_x is affectiing the results
    kmeans = KMeans(n_clusters=dataset_size, random_state=rank + 42, n_init=10)
    kmeans.fit(local_x.cpu().numpy().reshape(-1, 1))  # fit on CPU for KMeans
    # kmeans.fit(local_x_normalized.cpu().numpy().reshape(-1, 1))  # fit on CPU for KMeans

    local_pseudo_x = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    # local_pseudo_x = local_pseudo_x * x_std + x_mean # denormalize the pseudo dataset

    # print(f"Rank {rank} - Local pseudo dataset is :{local_pseudo_x}")
    # print(f"Rank {rank} - Local pseudo dataset size: {local_pseudo_x.size(0)}")




    model_sparse = SparseGPModel(local_pseudo_x).to(device)
    likelihood_sparse = gpytorch.likelihoods.GaussianLikelihood().to(device)
    mll_sparse = gpytorch.mlls.VariationalELBO(likelihood_sparse, model_sparse, num_data=local_x.size(0))

    optimizer_sparse = torch.optim.Adam( [{'params': model_sparse.parameters()},
                                        {'params': likelihood_sparse.parameters()}],
                                        lr=0.015,             
                                        betas=(0.9, 0.999),   # Default, but explicit for clarity
                                        eps=1e-8,             # Default
                                        weight_decay=1e-4,    # Add regularization
                                        amsgrad=True)         # Enable AMSGrad

    model_sparse.train()
    likelihood_sparse.train()

    # batch training
    batch_size= 64
    train_dataset = TensorDataset(local_x, local_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)  
            batch_y = batch_y.to(device) 
            
            optimizer_sparse.zero_grad()
            output = model_sparse(batch_x)
            loss = -mll_sparse(output, batch_y)
            penalty = inducing_penalty(model_sparse.variational_strategy.inducing_points, 
                                        x_min=x_min, x_max=x_max, margin=0.01)
            loss += 1.0* penalty
            loss.backward()
            optimizer_sparse.step()

        if rank == 0 and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.3f}")

        if loss.item() < 1e-6:
            if rank == 0:
                print(f"Converged at epoch {epoch + 1} with loss {loss.item():.3f}")
            break

    local_pseudo_x = model_sparse.variational_strategy.inducing_points.detach()
    local_pseudo_x = local_pseudo_x.to(device)
    
    # clear gradients
    optimizer_sparse.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    # evaluate local_pseudo_y using the local sparse GP model
    model_sparse.eval()
    likelihood_sparse.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        local_pseudo_y = likelihood_sparse(model_sparse(local_pseudo_x)).mean

    # local sparse GP hyperparameters
    local_hyperparams = torch.tensor([model_sparse.mean_module.constant.item(),
                                    model_sparse.covar_module.base_kernel.lengthscale.item(),
                                    model_sparse.covar_module.outputscale.item()], dtype=torch.float32, device=device)


    
    print(f"Rank {rank} and World Size {world_size} ")
    
    # test data
    test_x, _ = generate_test_data(num_samples=200, input_dim=1, rank=rank, world_size=world_size, 
                                partition='sequential')
    test_x = test_x.to(device)  
    torch.linspace(0, 1, 200).to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood_sparse(model_sparse(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    # plot the results
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(local_x.cpu().numpy(), local_y.cpu().numpy(), 'k*', label='Local Data')
    plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b-', label='Mean Prediction')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), color='blue', alpha=0.2, label='Confidence Interval')

    plt.plot(local_pseudo_x.cpu().numpy(), local_pseudo_y.cpu().numpy(), 'ro', label='Inducing Points')
    plt.title(f'Rank {rank} - Local Pseudo Dataset')
    plt.xlabel('Input')
    plt.ylabel('Output')

    # plt.xlim(-1, 1)
    # plt.ylim(-1.5, 1.5)

    plt.legend()
    plt.grid()
    plt.show()
    

    print(f"Rank {rank} - Local pseudo dataset size: {local_pseudo_x.size(0)}")
    
    return local_pseudo_x, local_pseudo_y, local_hyperparams



def create_augmented_dataset(local_x, local_y, device, world_size: int=1, rank: int=0, dataset_size: int=50, 
                                num_epochs: int = 100):
    """
    Create augmented dataset (D_c+) = local dataset (D_i) + global communication dataset (D_c)
    Args:
        local_x: Local training input data. (D_i)
        local_y: Local training output data. (D_i)
        world_size: Number of processes.
        rank: Current process rank.
        dataset_size: Size of the communication dataset to create.
        num_epochs: Number of training epochs for the local sparse GP model.
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
    
    local_pseudo_x, local_pseudo_y, local_hyperparams = create_local_pseudo_dataset(local_x, local_y,
                            device, dataset_size=dataset_size, rank=rank, num_epochs=num_epochs)
    
    # Step 2: gather local pseudo dataset from all processes and create global pseudo dataset
    world_size, rank = init_distributed_mode(backend=backend)

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



def train_model(model, likelihood, train_x, train_y, device, admm_params, backend='nccl'):
                 
    """
    Train the model using pxADMM optimizer
    Args:
        model: The Gaussian Process model to train.
        likelihood: The likelihood function for the model.
        train_x: Training input data.
        train_y: Training output data.
        device: Device to run the model on (CPU or GPU).
        admm_params: Dictionary containing ADMM parameters:
        - num_epochs: Number of training epochs.
        - rho: ADMM parameter for convergence.
        - lip: Lipschitz constant for the kernel function.
        - tol_abs: Absolute tolerance for convergence.
        - tol_rel: Relative tolerance for convergence
        backend: Distributed backend to use ('nccl', 'gloo', etc.).
    Returns:
        model: The trained Gaussian Process model.
        likelihood: The likelihood function for the model.

    1. Each agent will train its local sparse model with local dataset and find optimal inducing points.
    2. Each agent will share its local inducing points with central node (rank 0).
    3. Central node will create a global inducing points dataset by concatenating all local inducing points.
    4. Each agent will receive the global inducing points dataset from central node.
    5. Each agent will create augmented dataset using local dataset + global inducing points dataset.
            OR just use the global inducing points dataset as augmented dataset.
    6. Each agent will train its local model with augmented dataset again.
    """

    # Stage 1: Train local sparse GP model on local dataset and find optimal inducing points   
    # world_size, rank = init_distributed_mode(backend=backend)

    pseudo_x, pseudo_y, avg_hyperparams = create_augmented_dataset(train_x, train_y, device, world_size=world_size,
                            rank=rank, dataset_size=50, num_epochs=200)

    
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
    optimizer = pxadmm(
                        # [{'params': model.parameters()},
                        # {'params': likelihood.parameters()}],       
                        model.parameters(), 
                        rho=admm_params['rho'], lip=admm_params['lip'],
                            tol_abs=admm_params['tol_abs'], tol_rel=admm_params['tol_rel'],
                            rank=rank, world_size=world_size, backend=backend)
    
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

    num_epochs = admm_params['num_epochs']
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
    
    admm_params = {}
    admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lip'] = float(config.get('lip', 1.0))
    admm_params['tol_abs'] = float(config.get('tol_abs', 1e-6))
    admm_params['tol_rel'] = float(config.get('tol_rel', 1e-4))
    
    backend = str(config.get('backend', 'nccl'))

    # OS enviorment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # generate local training data
    local_x, local_y = generate_training_data(num_samples=num_samples, input_dim=input_dim, rank=rank, 
                                                world_size=world_size, partition='sequential')


    local_pseudo_x, local_pseudo_y, local_hyperparams = create_local_pseudo_dataset(local_x, local_y,
                            device, dataset_size=50, rank=rank, world_size=world_size, num_epochs=100)



    # # create the local model and likelihood
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()  
    # model = ExactGPModel(local_x, local_y, likelihood)

    # # train the model
    # model, likelihood, pseudo_x, pseudo_y = train_model(model, likelihood, local_x, local_y, device, 
    #                                         admm_params=admm_params, backend=backend)
    
    
    # # test the model
    # test_x = torch.linspace(0, 1, 1000)
    # mean, lower, upper = test_model(model, likelihood, test_x, device)

    # # plot the results
    # # plot_result(local_x, local_y, test_x, mean, lower, upper, rank=rank)
    # plot_result(pseudo_x, pseudo_y, test_x, mean, lower, upper, rank=rank)

