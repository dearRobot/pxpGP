import torch
import gpytorch
from matplotlib import pyplot as plt
import os

from utils import load_yaml_config, generate_training_data, generate_test_data
from utils.results import plot_result

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

    for epoch in range(optim_param['num_epochs']):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{optim_param["num_epochs"]}, Loss: {loss.item()}')

        if loss.item() < 0.2:
            print(f'Early stopping at epoch {epoch + 1}, Loss: {loss.item()}')
            break

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return model, likelihood
    

def test_model(model, likelihood, test_x, device):
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

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = test_x.to(device)
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

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
    
    optim_param = {}
    optim_param['num_epochs'] = int(config.get('num_epochs', 100))
    optim_param['lr'] = float(config.get('lr', 0.01))


    # generate training data
    train_x, train_y = generate_training_data(num_samples, input_dim, rank=0, world_size=1, 
                                                partition='random')

    # model and likelihood
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel)

    # train the model
    model, likelihood = train_model(model, likelihood, train_x, train_y, optim_param, device)

    # test the model
    test_x = generate_test_data(num_samples, input_dim, rank=0, world_size=1, partition='random')
    mean, lower, upper = test_model(model, likelihood, test_x, device)

    # generating results
    # save model and likelihood parameters
    # compute RMSE error 
    
    # plot the results
    plot_result(train_x, train_y, test_x, mean, lower, upper, input_dim=input_dim, rank=0)