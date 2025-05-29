import torch
import yaml

# TODO: make it multi-dimensional


def generate_training_data(num_samples, input_dim: int=1, rank: int=0, world_size: int=1, 
                           partition: str='random'):
    """
    Generate synthetic training data for Gaussian Processes.
    Args:
        num_samples: Number of samples to generate.
        input_dim: Dimensionality of the input data.
        rank: Current process rank (for distributed training).
        world_size: Total number of processes (for distributed training).
        partition: Method to partition the data among processes (default: 'random')
                    random: Randomly partition the data.
                    sequential: Sequentially partition the data.
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
    
    
    # dimensions
    if input_dim == 1:
        # f (x) = 5 x^2 sin(12x) + (x^3 − 0.5) sin(3x − 0.5)+ 4 cos(2x) + noise,
        train_x = torch.linspace(0, 1, num_samples)
        # train_y = 5 * train_x**2 * torch.sin(12*train_x) + (train_x**3 - 0.5) * torch.sin(3*train_x - 0.5) + 4 * torch.cos(2*train_x) + torch.randn(train_x.size()) * 0.2

        train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2


    elif input_dim == 2:
        print("Generating 2D training data is not yet implemented.")
    else:
        raise ValueError("Input dimension must be either 1 or 2 for this function.")
    
    
    if rank == 0:
        print("global train_x shape:", train_x.shape)
        print("global train_y shape:", train_y.shape)


    # partition criteria
    if partition == 'random':
        # divide dataset into m parts based on world size and rank
        torch.manual_seed(42) 
        local_indices = torch.randperm(train_x.size(0))
        split_indices = torch.chunk(local_indices, world_size)
        local_indices = split_indices[rank]
        local_x = train_x[local_indices]
        local_y = train_y[local_indices]

    elif partition == 'sequential':
        # divide dataset into m parts based on world size and rank
        local_size = num_samples // world_size
        start_idx = rank * local_size
        end_idx = start_idx + local_size if rank < world_size - 1 else num_samples
        local_x = train_x[start_idx:end_idx]
        local_y = train_y[start_idx:end_idx]

    else:
        raise ValueError("Invalid partition method. Use 'random' or 'sequential'.")


    if rank == 0:
        print("local_x shape:", local_x.shape)
        print("local_y shape:", local_y.shape)

    return local_x, local_y


def load_yaml_config(config_path):
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