import torch
import gpytorch
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time
import json
from filelock import FileLock

from utils import generate_dataset
from utils import load_yaml_config
from utils.results import plot_result, save_params

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
    

def train_model(model, likelihood, train_x, train_y, optim_param, device):
    """
    Train the Gaussian Process model using standard optimization.
    Args:
        model: The Gaussian Process model to train.
        likelihood: The likelihood function for the model.
        train_x: Training input data.
        train_y: Training output data.
        device: Device to run the training on (CPU or GPU).
    Returns:
        model: The trained Gaussian Process model.
        likelihood: The likelihood function for the model.
    """
    
    # move data to device
    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr= optim_param['lr'])

    model.train()
    likelihood.train()

    prev_loss = float('inf')
    mll_tol = 1e-4
    
    for epoch in range(optim_param['num_epochs']):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # relative mll change
            rel_change = abs(loss.item() - prev_loss) / (abs(prev_loss)  + 1e-8)
            prev_loss = loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{optim_param["num_epochs"]}, relative change: {rel_change:.4f}, Loss: {loss.item():.4f}')
            
            if rel_change < mll_tol:
                print(f'Early stopping at epoch {epoch + 1}, relative change: {rel_change:.4f}')
                break


    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return model, likelihood
    

def test_model(model, likelihood, test_x, test_y, device):
    """
    Test the Gaussian Process model on new data.
    Args:
        model: The trained Gaussian Process model.
        likelihood: The likelihood function for the model.
        test_x: Test input data.
        device: Device to run the testing on (CPU or GPU).
    Returns:
        mean: Predicted mean values.
        lower: Lower confidence interval.
        upper: Upper confidence interval.
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
    torch.sqrt(torch.mean((mean - test_y) ** 2)).item()
    print(f"RMSE: {torch.sqrt(torch.mean((mean - test_y) ** 2)).item():.4f}")
    
    return mean.cpu(), lower.cpu(), upper.cpu()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config_path = "config/fullGP.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    config = load_yaml_config(config_path)

    num_samples = int(config.get('num_samples', 1000))
    input_dim = int(config.get('input_dim', 1))
    test_split = float(config.get('test_split', 0.2))
    
    optim_param = {}
    optim_param['num_epochs'] = int(config.get('num_epochs', 100))
    optim_param['lr'] = float(config.get('lr', 0.01))

    # generate training data (train, validation, test)
    x, y = generate_dataset(num_samples, input_dim)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_split, random_state=42)

    print(f"Training data shape: {x.shape}, {y.shape}")
    print(f"Test data shape: {test_x.shape}, {test_y.shape}")

    # model and likelihood
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, lengthscale_constraint=gpytorch.constraints.Interval(0.01, 10.0))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0))
    model = ExactGPModel(train_x, train_y, likelihood, kernel)
    
    # train the model
    start_ = time.time()
    model, likelihood = train_model(model, likelihood, train_x, train_y, optim_param, device)
    train_time = time.time() - start_

    # test the model
    mean, lower, upper = test_model(model, likelihood, test_x, test_y, device)

    # print model and likelihood parameters
    if model.covar_module.base_kernel.lengthscale.numel() > 1:
        print("\033[92mLengthscale:\033[0m", model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()) 
    else:
        print("\033[92mLengthscale:\033[0m", model.covar_module.base_kernel.lengthscale.item()) 
        
    print("\033[92mOutputscale:\033[0m", model.covar_module.outputscale.item())
    print("\033[92mNoise:\033[0m", model.likelihood.noise.item())
    
    result={
        'model': 'fullGP',
        'rank': 0,
        'world_size': 1,
        'total_dataset_size': train_x.shape[0],
        'local_dataset_size': train_x.shape[0],
        'input_dim': input_dim,
        'lengthscale': model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().tolist(),
        'outputscale': model.covar_module.outputscale.item(),
        'noise': model.likelihood.noise.item(),
        'test_rmse': torch.sqrt(torch.mean((mean - test_y) ** 2)).item(),
        'train_time': train_time
        }
    
    file_path = f'results/dim_{input_dim}/result_dim{input_dim}_datasize_{x.shape[0]}.json'
    lock_path = file_path + '.lock'

    with FileLock(lock_path):
        with open(file_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

