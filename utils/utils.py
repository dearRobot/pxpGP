import torch
import yaml

# TODO: make it multi-dimensional

def generateTrainingData(num_samples, input_dim: int = 0, rank: int = 0, 
                         world_size: int = 1):
    """
    Generate synthetic training data for Gaussian Processes.
    Args:
        num_samples: Number of samples to generate.
        input_dim: Dimensionality of the input data.
        rank: Current process rank (for distributed training).
        world_size: Total number of processes (for distributed training).
    Returns:
        local_x: Generated local input data.
        local_y: Generated local output data.
    """
    if input_dim <= 0:
        raise ValueError("Input dimension must be greater than 0.")
    if num_samples <= 0:
        raise ValueError("Number of samples must be greater than 0.")
    if world_size <= 0:
        raise ValueError("World size must be greater than 0.")
    if rank < 0 or rank >= world_size:
        raise ValueError("Rank must be between 0 and world_size - 1.")
    
    # Generate synthetic data
    train_x = torch.linspace(0, 1, num_samples)
    train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2

    if rank == 0:
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

    return local_x, local_y


def loadYAMLConfig(config_path):
    """
    Load a YAML configuration file.
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        config: Loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config