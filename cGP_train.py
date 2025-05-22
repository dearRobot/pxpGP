import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import cadmm
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

# global GP model used for prediction as a combination of local models
class GlobalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, global_covar_inv):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.global_covar_inv = global_covar_inv
    
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


def train_model(model, likelihood, train_x, train_y, num_epochs=10, backend='nccl'):
    # Initialize distributed training
    world_size, rank = init_distributed_mode(backend=backend)
    
    # move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # # split dataset 
    # dataset = TensorDataset(train_x, train_y)
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # train_loader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # optimizer
    optimizer = cadmm(model.parameters(), lr=0.005, max_iter=10, rho=0.85, rank=rank, world_size=world_size)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -model.likelihood(output).log_prob(train_y)
        return loss, output
    
    model.train()
    likelihood.train()

    for epoch in range(num_epochs):
        optimizer.step(closure)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {closure()[0].item()}")  


    # verification of model hyperparameters
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu()
    outputscale = model.covar_module.outputscale.detach().cpu()
    noise = model.likelihood.noise.detach().cpu()

    print(f"Rank {rank}: Lengthscale: {lengthscale}, Outputscale: {outputscale}, Noise: {noise}")

    # compute local inverse covariance matrix
    # with torch.no_grad():
    #     model.eval()
    #     likelihood.eval()
    #     local_covar = model.covar_module(train_x).evaluate()
    #     local_covar += (likelihood.noise) * torch.eye(local_covar.size(0), device=device) # K + \sigma^2I
    #     local_covar_inv = torch.inverse(local_covar).contiguous() # K^{-1}

    # print("Rank {}: local_covar_inv size: {}".format(rank, local_covar_inv.size()))
    
    # PyTorch’s distributed communication functions (dist.send, dist.recv, dist.all_gather) require tensors to be contiguous
    
    # synchronize x parameter why? each agent updates its local x (model parameters) independently, but 
    # z (auxiliary variable) is synchronized. Averaging x ensures the global model uses a consensus set 
    # of parameters, aligning the kernel function across agents.
    # optimizer.synchronize_parameters()
    
    # gather local inverse covariance matrix sizes
    # local_covar_inv_size = torch.tensor(local_covar_inv.size(0), device=device)
    # all_covar_inv_sizes = [torch.tensor(0, device=device) for _ in range(world_size)] # list of tensors to store the sizes from all agents.
    # dist.all_gather(all_covar_inv_sizes, local_covar_inv_size)

    # dist.barrier()  # Synchronize before send/receive
    
    # # form global covariance matrix 
    # total_size = sum(size.item() for size in all_covar_inv_sizes)
    # global_covar_inv = torch.zeros(total_size, total_size, device=device)

    # if rank == 0:
    #     # Collects the local inverted covariance matrices
    #     offset = 0
    #     for i, size in enumerate(all_covar_inv_sizes):
    #         size = size.item()
    #         print("Rank {}: local_covar_inv size: {}".format(i, size))
    #         if i == 0: # for rank 0
    #             global_covar_inv[offset:offset+size, offset:offset+size] = local_covar_inv
    #         else:
    #             temp = torch.zeros(size, size, device=device)
    #             print("Rank {}: temp size: {}".format(i, temp.size()))
                
    #             # dist.recv(temp, src=i)
    #             req = dist.irecv(temp, src=i)
    #             req.wait()  # Wait for receive to complete
                
    #             print("Rank {}: temp size after recv: {}".format(i, temp.size()))
    #             global_covar_inv[offset:offset+size, offset:offset+size] = temp
    #         offset += size
    # else:
    #     # send local inverted covariance matrix to rank 0
    #     print(f"Rank {rank}: Sending local_covar_inv size {local_covar_inv.size()} to rank 0")
    #     req = dist.isend(local_covar_inv, dst=0)
    #     req.wait()  # Wait for send to complete
    #     # dist.send(local_covar_inv, dst=0)
    #     print(f"Rank {rank}: Completed sending local_covar_inv to rank 0")
        
    # dist.barrier() # synchronize all processes. Ensures all processes wait until the global model is created before returning. 

    # global covariance matrix = block diagonal matrix(local covariance matrices)
    # gather local training data, not required for now centralized training but useful for decentralized and pseudo-distributed training
    # local_x_size = torch.tensor(local_x.size(0), device=device)
    # all_x_sizes = [torch.tensor(0, device=device) for _ in range(world_size)]
    # dist.all_gather(all_x_sizes, local_x_size)

    # if rank == 0:
    #     print("Global covariance matrix size:", global_covar_inv.size())
    #     print("Global covariance matrix:", global_covar_inv)
    
    dist.destroy_process_group()
    return model, likelihood

def test_model(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = test_x.to(device)
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    return mean, lower, upper

def plot_results(train_x, train_y, test_x, mean, lower, upper):
    plt.figure(figsize=(12, 6))
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Train Data')
    plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b', label='Mean Prediction')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, color='blue', label='Confidence Interval')
    plt.title('Gaussian Process Regression Rank {}'.format(os.environ['RANK']))
    plt.legend()
    plt.show()


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

    # Create the local model and likelihood    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(local_x, local_y, likelihood)

    # os environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Train the local model
    model, likelihood = train_model(model, likelihood, local_x, local_y, num_epochs=5, backend='gloo')

    # Test the model
    test_x = torch.linspace(0, 1, 51)
    mean, lower, upper = test_model(model, likelihood, test_x)

    # Plot the local results
    plot_results(local_x, local_y, test_x, mean, lower, upper)



# Rank 0: Lengthscale: tensor([[0.3729]]), Outputscale: 2.439328193664551, Noise: tensor([0.1141])

# Rank 0: Lengthscale: tensor([[0.3437]]), Outputscale: 2.160946846008301, Noise: tensor([0.0390])
# Rank 1: Lengthscale: tensor([[0.3378]]), Outputscale: 2.20019268989563, Noise: tensor([0.0398])

# Rank 2: Lengthscale: tensor([[0.3260]]), Outputscale: 1.8940331935882568, Noise: tensor([0.0392])
# Rank 1: Lengthscale: tensor([[0.3166]]), Outputscale: 1.8676905632019043, Noise: tensor([0.0400])
# Rank 0: Lengthscale: tensor([[0.3379]]), Outputscale: 1.8880828619003296, Noise: tensor([0.0389])


