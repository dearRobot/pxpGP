#!/usr/bin/env python3

import torch
import yaml

# TODO: make it multi-dimensional


# def get_partioned_data(data, world_size: int=1, rank: int=0, partition: str='random'):

# def generate_1d_data(num_samples, rank: int=0, world_size: int=1, partition: str='random'):

# def generate_2d_data(num_samples, rank: int=0, world_size: int=1, partition: str='random'):

# def generate_3d_data(num_samples, rank: int=0, world_size: int=1, partition: str='random'):


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
        # f(x) = 5 x^2 sin(12x) + (x^3 − 0.5) sin(3x − 0.5)+ 4 cos(2x) + noise,
        if rank == 0:
            print("Generating synthetic 1D training data...")
        
        train_x = torch.linspace(0, 1, num_samples)
        train_y = 5 * train_x**2 * torch.sin(12*train_x) + (train_x**3 - 0.5) * torch.sin(3*train_x - 0.5) + 4 * torch.cos(2*train_x) + torch.randn(train_x.size()) * 0.2


    elif input_dim == 2:
        # f(X) = (1 + ((x1 + x2 + 1)**2) * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) *
        #         (30 + ((2*x1 - 3*x2)**2) * (18 - 32*x1 + 12*x1**2 + 12*x2 - 36*x1*x2 + 27*x2**2))
        if rank == 0:
            print("Generating Goldstein-Price 2D training data...")

        train_x = torch.rand(num_samples, 2)  # Randomly generate 2D points in [0, 1]^2
        
        train_y = (100 * (train_x[:, 1] - train_x[:, 0]**2)**2 + (1 - train_x[:, 0])**2) 
        # train_y = (1 + ((train_x[:, 0] + train_x[:, 1] + 1)**2) * (19 - 14 * train_x[:, 0] + 3 * train_x[:, 0]**2 - 14 * train_x[:, 1] + 6 * train_x[:, 0] * train_x[:, 1] + 3 * train_x[:, 1]**2)) * \
                #  (30 + ((2 * train_x[:, 0] - 3 * train_x[:, 1])**2) * (18 - 32 * train_x[:, 0] + 12 * train_x[:, 0]**2 + 12 * train_x[:, 1] - 36 * train_x[:, 0] * train_x[:, 1] + 27 * train_x[:, 1]**2))
        
        train_y += torch.randn(train_x.size(0)) * 0.2  # Add noise to the output
        

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

    elif partition == 'full':
        # No partitioning, return the full dataset
        local_x = train_x
        local_y = train_y

    else:
        raise ValueError("Invalid partition method. Use 'random' or 'sequential'.")


    if rank == 0:
        print("local_x shape:", local_x.shape)
        print("local_y shape:", local_y.shape)

    return local_x, local_y


def generate_test_data(num_samples, input_dim: int=1, rank: int=0, world_size: int=1,
                       partition: str='random'):
    """
    Generate synthetic test data for Gaussian Processes.
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
        test_x = torch.linspace(0, 1, num_samples)
        
    elif input_dim == 2:
        test_x = torch.rand(num_samples, 2)  
        # x1 = torch.linspace(-1, 1, num_samples)
        # x2 = torch.linspace(-1, 1, num_samples)
        # x1, x2 = torch.meshgrid(x1, x2)
        # test_x = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    
    
    else:
        raise ValueError("Input dimension must be either 1 or 2 for this function.")

    
    return test_x
    
    # # partition criteria
    # if partition == 'random':
    #     # divide dataset into m parts based on world size and rank
    #     torch.manual_seed(42) 
    #     local_indices = torch.randperm(test_x.size(0))
    #     split_indices = torch.chunk(local_indices, world_size)
    #     local_indices = split_indices[rank]
    #     local_x = test_x[local_indices]
    #     local_y = test_y[local_indices]
        

    # elif partition == 'sequential':
    #     # divide dataset into m parts based on world size and rank
    #     local_size = num_samples // world_size
    #     start_idx = rank * local_size
    #     end_idx = start_idx + local_size if rank < world_size - 1 else num_samples
    #     local_x = test_x[start_idx:end_idx]
    #     local_y = test_y[start_idx:end_idx]

    # elif partition == 'full':
    #     # No partitioning, return the full dataset
    #     local_x = test_x
    #     local_y = test_y

    # else:
    #     raise ValueError("Invalid partition method. Use 'random' or 'sequential'.")
    
    # if rank == 0:
    #     print("local_x shape:", local_x.shape)
    #     print("local_y shape:", local_y.shape)

    # return local_x, local_y


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