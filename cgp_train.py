import torch
import gpytorch
from matplotlib import pyplot as plt
from admm import cadmm
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os


# GP Model
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

    print(f"Initialized distributed mode with world size {world_size} and rank {dist.get_rank()}")
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

    # split dataset 
    dataset = TensorDataset(train_x, train_y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # optimizer
    optimizer = cadmm(model.parameters(), lr=0.005, max_iter=10, rho=0.85, rank=rank, world_size=world_size)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -model.likelihood(output).log_prob(train_y)
        # loss.backward()
        return loss, output

    model.train()
    likelihood.train()

    for epoch in range(num_epochs):
        # print(f"Epoch {epoch+1}/{num_epochs}")
        sampler.set_epoch(epoch)
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.step(closure)

        # Print loss
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {closure()[0].item()}")  


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
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Starting training...")
    
    world_size = 1

    # 1D Data
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2

    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # os environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = '0'

    # Train the model
    model, likelihood = train_model(model, likelihood, train_x, train_y, num_epochs=5, backend='nccl')

    # Test the model
    test_x = torch.linspace(0, 1, 51)
    mean, lower, upper = test_model(model, likelihood, test_x)

    # Plot the results
    plot_results(train_x, train_y, test_x, mean, lower, upper)
