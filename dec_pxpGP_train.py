import torch
import gpytorch
from admm import pxadmm, dec_pxadmm
from admm import scaled_pxadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
from linear_operator.settings import max_cg_iterations, cg_tolerance
import time
from filelock import FileLock
import json

from utils import load_yaml_config
from utils.results import plot_result
from utils import generate_dataset, split_agent_data
from utils.graph import DecentralizedNetwork
from utils.results import save_params

from sklearn.cluster import KMeans

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
    

# sparse GP Model with inducing or pseudo points/variational distribution
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, 
                                    variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
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


def replusive_penalty(inducing_x, min_dist: float=0.01, input_dim: int=1):
    """
    Compute the repulsive penalty for inducing points.
    Args:
        inducing_x: Inducing points tensor.
        min_dist: Minimum distance between inducing points.
    Returns:
        penalty: Repulsive penalty value.
    """
    n_points = inducing_x.size(0)
    if n_points < 2:
        return torch.tensor(0.0, device=inducing_x.device)
    
    distances = torch.cdist(inducing_x, inducing_x)
    mask = torch.triu(torch.ones(n_points, n_points), diagonal=1).bool()
    close_distances = distances[mask] - min_dist
    penalty = torch.relu(-close_distances).pow(2).sum()
    return penalty


def boundary_penalty(inducing_x, x_min: float=0.0, x_max: float=1.0, margin=0.01):
    below = torch.relu(x_min - inducing_x + margin)
    above = torch.relu(inducing_x - x_max + margin)
    return (below**2 + above**2).sum()


def create_local_pseudo_dataset(local_x, local_y, device, dataset_size: int=50, world_size: int=1, 
                                rank: int=0, num_epochs: int=100, input_dim: int=1):
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
        
    x_min = local_x.min().item()
    x_max = local_x.max().item()
    
    if rank == 0:
        print(f"\033[92mRank {rank} - sparse dataset size is: {dataset_size}, local dataset: {local_x.shape}, \033[0m")
    
    kmeans = KMeans(n_clusters=dataset_size, random_state=rank + 42, n_init=10)
    
    if input_dim == 1:
        kmeans.fit(local_x.cpu().numpy().reshape(-1, 1)) 
    else:
        kmeans.fit(local_x.cpu().numpy())

    local_pseudo_x = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    model_sparse = SparseGPModel(local_pseudo_x, kernel).to(device)
    likelihood_sparse = gpytorch.likelihoods.GaussianLikelihood().to(device)
    mll_sparse = gpytorch.mlls.VariationalELBO(likelihood_sparse, model_sparse, num_data=local_x.size(0))

    # torch.optim.LBFGS
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
    # batch_size= 64
    batch_size = min(int(local_x.size(0) / 10), 50)
    train_dataset = TensorDataset(local_x, local_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    prev_loss = float('inf')
    elbo_tol = 1e-4

    if rank == 0:
        print(f"\033[92mRank {rank} - Training local sparse GP model with {local_x.size(0)} samples\033[0m")

    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)  
            batch_y = batch_y.to(device) 

            # print(f"Rank {rank} - batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
            
            optimizer_sparse.zero_grad()
            output = model_sparse(batch_x)
            loss = -mll_sparse(output, batch_y)
            b_penalty = boundary_penalty(model_sparse.variational_strategy.inducing_points, 
                                        x_min=x_min, x_max=x_max, margin=0.0)
            r_penalty = replusive_penalty(model_sparse.variational_strategy.inducing_points,
                                        min_dist=0.02, input_dim=input_dim)
            loss += 10.0* b_penalty + 1.0 * r_penalty
            loss.backward()
            optimizer_sparse.step()

        if rank == 0 and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.3f}")

        if epoch % 5 == 0:
            # relative change in loss
            rel_change = abs(loss.item() - prev_loss) / (abs(prev_loss) + 1e-8)
            prev_loss = loss.item()

            if rel_change < elbo_tol:
                if rank == 0:
                    print(f'Rank {rank} Early stopping at epoch {epoch + 1}, relative change: {rel_change:.4f}')
                break
        
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

    # need to modify for multi-dimensional input
    # local sparse GP hyperparameters

    mean_const = model_sparse.mean_module.constant.detach().view(-1)                  # shape [1]
    lengthscale = model_sparse.covar_module.base_kernel.lengthscale.detach().view(-1) # shape [D]
    outputscale = model_sparse.covar_module.outputscale.detach().view(-1) 
    
    local_hyperparams = torch.cat([mean_const, lengthscale, outputscale]) 
        
    return local_pseudo_x, local_pseudo_y, local_hyperparams


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


def create_augmented_dataset(local_x, local_y, device, neighbors, world_size: int=1, rank: int=0, dataset_size: int=50, 
                                num_epochs: int = 100, input_dim: int=1, backend='nccl'):
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
    
    #  Step 1: create local pseudo dataset    
    local_x = local_x.to(device)
    local_y = local_y.to(device)

    dataset_size = min(int(local_x.size(0) / world_size),  int(local_x.size(0) / 10))
    dataset_size = max(dataset_size, 5)

    local_pseudo_x, local_pseudo_y, local_hyperparams = create_local_pseudo_dataset(local_x, local_y,
                            device, dataset_size=dataset_size, rank=rank, num_epochs=num_epochs, 
                            input_dim=input_dim)
    
    # Step 2: gather local pseudo dataset from all processes and create global pseudo dataset by flooding
    # right now for testing we will just simply send to all
# TODO: implement flooding, gossip, etc.
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12345')
    world_size, rank = init_distributed_mode(backend=backend, master_addr=master_addr, 
                                              master_port=master_port)
    
    
    # broadcast_data(local_pseudo_x, neighbors, rank=rank, world_size=world_size)
    # broadcast_data(local_pseudo_y, neighbors, rank=rank, world_size=world_size)
    # broadcast_data(local_hyperparams, neighbors, rank=rank, world_size=world_size)
    
    sample_x_list = [torch.empty_like(local_pseudo_x) for _ in range(world_size)]
    sample_y_list = [torch.empty_like(local_pseudo_y) for _ in range(world_size)]

    dist.gather(local_pseudo_x, gather_list=sample_x_list if rank == 0 else None, dst=0)
    dist.gather(local_pseudo_y, gather_list=sample_y_list if rank == 0 else None, dst=0)

    if rank == 0:
        comm_x = torch.cat(sample_x_list, dim=0)
        comm_y = torch.cat(sample_y_list, dim=0)
    else:
        comm_x = torch.zeros((dataset_size * world_size, input_dim), dtype=local_pseudo_x.dtype, device=device)
        comm_y = torch.zeros(dataset_size * world_size, dtype=local_y.dtype, device=device)

    if input_dim == 1:
        comm_x = comm_x.squeeze(-1)
        comm_y = comm_y.squeeze(-1)
    
    # broadcast the communication dataset to all agents from rank 0
    dist.broadcast(comm_x, src=0)
    dist.broadcast(comm_y, src=0)

    # create augmented dataset
    pseudo_x = torch.cat([local_x, comm_x], dim=0)
    pseudo_y = torch.cat([local_y, comm_y], dim=0)
    

    # Step 3: Share the local model hyperparameters with other agents same as local_pseudo_x using flooding
# TODO: implement flooding, gossip, etc.
    hyperparams_list = [torch.empty_like(local_hyperparams) for _ in range(world_size)] 
    dist.gather(local_hyperparams, gather_list=hyperparams_list if rank == 0 else None, dst=0)

    if rank == 0:   
        hyperparam_stack = torch.stack(hyperparams_list)
        avg_hyperparams_ = hyperparam_stack.mean(dim=0)
    else:
        avg_hyperparams_ = torch.zeros_like(local_hyperparams, dtype=torch.float32, device=device)

    dist.broadcast(avg_hyperparams_, src=0)

    # need to modify for multi-dimensional input
    avg_hyperparams = {'mean_constant': avg_hyperparams_[0].item(),
                        'lengthscale': avg_hyperparams_[1].item(),
                        'outputscale': avg_hyperparams_[2].item()}
    
    torch.cuda.empty_cache()
    return pseudo_x, pseudo_y, avg_hyperparams
    

def train_model(train_x, train_y, device, admm_params, neighbors, input_dim: int=1, backend='nccl'):
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
        neighbors: List of neighboring agents.
        input_dim: Input dimension of the data.
    Returns:
        Trained model and likelihood.
        pseudo_x: Pseudo points for the model.
        pseudo_y: Pseudo points for the model.

    1. Each agent will train its local sparse model with local dataset and find optimal inducing points.
    2. Each agent will share its local inducing points with all agents via flooding.
    3. Each agent will create its local pseudo dataset using local dataset + global communication dataset
    4. Share the local hyperparameters with other agents and create global hyperparameters
    5. Each agent will train its local model with augmented dataset again. (with z consensus update via DAC)
    6. Each agent will update its local model with global hyperparameters
    """
    
    # Stage 1: Train local sparse GP model on local dataset and find optimal inducing points   
    pseudo_x, pseudo_y, avg_hyperparams = create_augmented_dataset(train_x, train_y, device, neighbors=neighbors, world_size=world_size,
                            rank=rank, dataset_size=50, num_epochs=admm_params['num_epochs'], input_dim=input_dim, backend=backend)

    if rank == 0:
        print(f"Rank {rank} - Augmented dataset size: {pseudo_x.size(0)}")
    
    # Stage 2: Train on augmented dataset with warm start
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(pseudo_x, pseudo_y, likelihood, kernel)

    # warm start
    model.mean_module.constant.data = avg_hyperparams['mean_constant'] * torch.ones_like(model.mean_module.constant.data)
    model.covar_module.base_kernel.lengthscale.data = avg_hyperparams['lengthscale'] * torch.ones_like(model.covar_module.base_kernel.lengthscale.data)
    model.covar_module.outputscale.data = avg_hyperparams['outputscale'] * torch.ones_like(model.covar_module.outputscale.data)
    
    model = model.to(device)
    likelihood = likelihood.to(device)
    pseudo_x = pseudo_x.to(device)
    pseudo_y = pseudo_y.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = dec_pxadmm(model.parameters(), neighbors=neighbors, rho=admm_params['rho'],
                            lip=admm_params['lip'], rank=rank, world_size=world_size, 
                            tol_abs=admm_params['tol_abs'], tol_rel=admm_params['tol_rel'])
    
    def closure():
        optimizer.zero_grad()
        with gpytorch.settings.min_preconditioning_size(0.005), max_cg_iterations(2000), cg_tolerance(1e-2):
            output = model(pseudo_x)
            loss = -mll(output, pseudo_y)
            loss.backward()
            grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                              for p in model.parameters()])
        return loss, grad
    
    model.train()
    likelihood.train()

    if rank == 0:
        print(f"\033[92mRank {rank} - Training global model with dec-pxADMM optimizer\033[0m")
    
    for epoch in range(admm_params['num_epochs']):
        converged = optimizer.step(closure, epoch=epoch)
        loss_val = closure()[0].item()
        
        # if rank == 0 and (epoch % 10 == 0 or epoch == admm_params['num_epochs'] - 1):
        #     print(f"Epoch {epoch+1}/{admm_params['num_epochs']}, Loss: {closure()[0].item()}")

        if not torch.isfinite(torch.tensor(loss_val)):
            if rank == 0:
                print(f"Epoch {epoch + 1}: Loss is NaN, stopping early.")
            break

        if converged:
            # if rank == 0:
            print(f"Rank {rank} - Converged at epoch {epoch + 1}, loss: {loss_val:.4f}")
            break

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()    

    dist.destroy_process_group()
    return model, likelihood, pseudo_x, pseudo_y


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
    print(f"Rank {rank} - Testing RMSE: {rmse_error:.4f}")
    
    return mean.cpu(), lower.cpu(), upper.cpu(), rmse_error


if __name__ == "__main__":
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config_path = 'config/dec_pxpGP.yaml'
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    test_split = float(config.get('test_split', 0.2))

    admm_params = {}
    admm_params['num_epochs'] = int(config.get('num_epochs', 100))
    admm_params['rho'] = float(config.get('rho', 0.8))
    admm_params['lip'] = float(config.get('lip', 1.0))
    admm_params['tol_abs'] = float(config.get('tol_abs', 1e-6))
    admm_params['tol_rel'] = float(config.get('tol_rel', 1e-4))

    backend = str(config.get('backend', 'nccl'))
    graph_viz = bool(config.get('graph_viz', False))
    aug_dataset_size = int(config.get('aug_dataset_size', 50))

    # generate local training data
    x, y = generate_dataset(num_samples, input_dim)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_split, random_state=42)

    # split data among agents
    local_x, local_y = split_agent_data(x, y, world_size, rank, input_dim=input_dim, partition='sequential')    
        
    # get information about neighbors
    dec_graph = DecentralizedNetwork(num_nodes=world_size, graph_type='degree', dim=input_dim, degree=2, seed=42)
    neighbors = dec_graph.neighbors[rank]
    print(f"Rank {rank} neighbors: {neighbors}")

    if graph_viz and rank == 0:
        dec_graph.visualize_graph()

    # Train the model
    start_time = time.time()
    model, likelihood, pseudo_x, pseudo_y  = train_model(local_x, local_y, device, admm_params, 
                                                         neighbors, input_dim=input_dim, backend=backend)
    train_time = time.time() - start_time

    # test the model
    mean, lower, upper, rmse_error = test_model(model, likelihood, test_x, test_y, device)

    # print model and likelihood parameters
    if model.covar_module.base_kernel.lengthscale.numel() > 1:
        print(f"\033[92mRank: {rank}, Lengthscale:", model.covar_module.base_kernel.lengthscale.cpu().detach().numpy(), "\033[0m")  # Print all lengthscale values
    else:
        print(f"\033[92mRank: {rank}, Lengthscale:", model.covar_module.base_kernel.lengthscale.item(), "\033[0m")  # Print single lengthscale value
    
    print(f"\033[92mRank: {rank}, Outputscale:", model.covar_module.outputscale.item(), "\033[0m")
    print(f"\033[92mRank: {rank}, Noise:", model.likelihood.noise.item(), "\033[0m")
    
    result={
        'model': 'decpxpGP',
        'rank': rank,
        'world_size': world_size,
        'total_dataset_size': x.shape[0],
        'local_dataset_size': local_x.shape[0],
        'input_dim': input_dim,
        'lengthscale': model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().tolist(),
        'outputscale': model.covar_module.outputscale.item(),
        'noise': model.likelihood.noise.item(),
        'test_rmse': rmse_error,
        'train_time': train_time
    }

    file_path = f'results/dim_{input_dim}/dec_result_dim{input_dim}_agents_{world_size}_datasize_{x.shape[0]}.json'
    lock_path = file_path + '.lock'
    
    with FileLock(lock_path):
        with open(file_path, 'a') as f:
            f.write(json.dumps(result) + '\n')