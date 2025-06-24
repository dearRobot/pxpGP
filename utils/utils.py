#!/usr/bin/env python3

import torch
import yaml
import numpy as np
import math


# TODO: make it multi-dimensional


# def get_partioned_data(data, world_size: int=1, rank: int=0, partition: str='random'):

# def generate_1d_data(num_samples, rank: int=0, world_size: int=1, partition: str='random'):

def generate_2d_data(num_samples, input_dim: int=2):
    # # Rosenbrock Function f(x) = (100 * (x2 - x1**2)**2 + (1 - x1)**2) + noise
    # train_x_np = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, input_dim))
    # train_y_np = (100 * (train_x_np[:, 1] - train_x_np[:, 0]**2)**2 + (1 - train_x_np[:, 0])**2)

    # train_x = torch.tensor(train_x_np, dtype=torch.float32)
    # train_y = torch.tensor(train_y_np, dtype=torch.float32) + torch.randn(train_x.size(0)) * 0.2

    # logarithmic form of the Goldstein-Price function, on [0, 1]
    # f(x) = log(1 + ((x1 + x2 + 1)**2) * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) * log(30 + ((2*x1 - 3*x2)**2) * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))

    train_x_np = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, 2))

    x1 = 4.0 * train_x_np[:, 0] - 2.0
    x2 = 4.0 * train_x_np[:, 1] - 2.0

    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2 = 30 + fact2a * fact2b

    prod = fact1 * fact2

    train_y_np = (np.log(prod) - 8.693) / 2.427

    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    
    return train_x, train_y

def generate_3d_data(num_samples):
    # Hartmann 3-Dimensional Function
    train_x_np = np.random.uniform(low=0.0, high=1.0, size=(num_samples, 3))
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 1e-4 * np.array([[3689.0, 1170.0, 2673.0],
                         [4699.0, 4387.0, 7470.0],
                         [1091.0, 8732.0, 5547.0],
                         [381.0, 5743.0, 8828.0]])
    
    x = np.atleast_2d(train_x_np)
    total = np.zeros(x.shape[0])
    
    for i in range(4):
        inner = np.sum(A[i, :] * (x - P[i, :])**2, axis=1)
        total += alpha[i] * np.exp(-inner)

    train_y_np = -total

    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32) + torch.randn(train_x.size(0)) * 0.2

    return train_x, train_y


def generate_dataset(num_samples, input_dim: int=1):
    if input_dim <= 0:
        raise ValueError("Input dimension must be greater than 0.")
    if num_samples <= 0:
        raise ValueError("Number of samples must be greater than 0.")
    
    # dimensions
    if input_dim == 1:
        # f(x) = 5 x^2 sin(12x) + (x^3 − 0.5) sin(3x − 0.5)+ 4 cos(2x) + noise,
        train_x = torch.linspace(0, 1, num_samples)
        train_y = 5 * train_x**2 * torch.sin(12*train_x) + (train_x**3 - 0.5) * torch.sin(3*train_x - 0.5) + 4 * torch.cos(2*train_x) 
        train_y += torch.randn(train_x.size()) * 0.2  # Add noise to the output
        return train_x, train_y

    elif input_dim == 2:
        train_x, train_y = generate_2d_data(num_samples)
        return train_x, train_y
    
    elif input_dim == 3:
        train_x, train_y = generate_3d_data(num_samples)
        return train_x, train_y
    
    else:
        raise ValueError("Input dimension must be either 1 or 2 for this function.")


def _kd_bisect(indices, pts, target_cells):
    """
    Recursively split `indices` (a 1-D LongTensor of row indices into `pts`)
    until we have `target_cells` disjoint index subsets.  Each split is along
    the longest side of the current bounding box at the median.
    """
    cells = [indices]
    while len(cells) < target_cells:
        # pick the largest cell so we don't create imbalanced tiny pieces
        big_idx = max(range(len(cells)), key=lambda i: cells[i].numel())
        big_cell = cells.pop(big_idx)
        cell_pts = pts[big_cell]

        # find longest dimension of this cell
        ranges = cell_pts.max(dim=0).values - cell_pts.min(dim=0).values
        split_dim = torch.argmax(ranges).item()

        # median along that dimension
        median_val = cell_pts[:, split_dim].median()
        left_mask = cell_pts[:, split_dim] <= median_val

        # make sure neither side becomes empty; if so nudge the threshold
        if left_mask.all() or (~left_mask).all():
            median_val = cell_pts[:, split_dim].mean()
            left_mask = cell_pts[:, split_dim] <= median_val

        cells.insert(big_idx, big_cell[left_mask])
        cells.append(big_cell[~left_mask])
    return cells


def _regular_grid_split(train_x, world_size, rank):
    """Return boolean mask of rows belonging to this rank."""
    N, d = train_x.shape
    # Number of cells per dimension (must be integer)
    cells_per_dim = round(world_size ** (1 / d))
    if cells_per_dim ** d != world_size:
        if rank == 0:
            print(f"\033[91mWarning: world_size={world_size} is not a perfect {d}-th power. Using regular grid split instead, which may lead to imbalanced partitions.\033[0m")
        return None, False
    
    # Map rank → tuple of cell indices (i0, i1, …, id-1)
    base   = cells_per_dim
    digits = []
    r = rank
    for _ in range(d):
        digits.append(r % base)
        r //= base
    digits = digits[::-1]          # most-significant first

    mask = torch.ones(N, dtype=torch.bool, device=train_x.device)
    for j, ij in enumerate(digits):
        low  = train_x[:, j].min()
        high = train_x[:, j].max()
        edges = torch.linspace(low, high, cells_per_dim + 1, device=train_x.device)
        mask &= (train_x[:, j] >= edges[ij]) & (train_x[:, j] <= edges[ij + 1])

    return mask, True


def split_agent_data(train_x, train_y, world_size: int=1, rank: int=0, input_dim: int=1, 
                     partition: str='random'):
    """
    Split the training data among multiple agents.
    Args:
        train_x: Input training data.
        train_y: Output training data.
        world_size: Total number of processes (for distributed training).
        rank: Current process rank (for distributed training).
        partition: Method to partition the data among processes (default: 'random')
                    random: Randomly partition the data.
                    sequential: Sequentially partition the data.
    Returns:
        local_x: Local input data for the current agent.
        local_y: Local output data for the current agent.
    """
    if world_size <= 0:
        raise ValueError("World size must be greater than 0.")
    if rank < 0 or rank >= world_size:
        raise ValueError("Rank must be between 0 and world_size - 1.")
    
    # partition criteria
    if partition == 'random':
        # divide dataset into m parts based on world size and rank
        # torch.manual_seed(42) 
        local_indices = torch.randperm(train_x.size(0))
        split_indices = torch.chunk(local_indices, world_size)
        
        local_indices = split_indices[rank]
        
        local_x = train_x[local_indices]
        local_y = train_y[local_indices]

    elif partition == 'sequential' and input_dim == 1:
        # divide dataset into m parts based on world size and rank
        local_size = train_x.size(0) // world_size
        start_idx = rank * local_size
        end_idx = start_idx + local_size if rank < world_size - 1 else train_x.size(0)
        local_x = train_x[start_idx:end_idx]
        local_y = train_y[start_idx:end_idx]

    elif partition == 'sequential':
        # if train_x.dim() != 2 or train_x.size(1) != input_dim:
        #     raise ValueError('train_x must be (N, input_dim) for spatial split')
        # all_idx   = torch.arange(train_x.size(0))
        # cells     = _kd_bisect(all_idx, train_x, world_size)
        # local_idx = cells[rank]
        mask, success = _regular_grid_split(train_x, world_size, rank)
        
        if success:
            local_idx = torch.nonzero(mask, as_tuple=False).squeeze()
        else:
            if train_x.dim() != 2 or train_x.size(1) != input_dim:
                raise ValueError('train_x must be (N, input_dim) for spatial split')
            all_idx   = torch.arange(train_x.size(0))
            cells     = _kd_bisect(all_idx, train_x, world_size)
            local_idx = cells[rank]
        
        local_x   = train_x[local_idx]
        local_y   = train_y[local_idx]
    
    else:
        raise ValueError("Invalid partition method. Use 'random' or 'sequential'.")

    return local_x, local_y



# def generate_training_data(num_samples, input_dim: int=1, rank: int=0, world_size: int=1, 
#                            partition: str='random'):
#     """
#     Generate synthetic training data for Gaussian Processes.
#     Args:
#         num_samples: Number of samples to generate.
#         input_dim: Dimensionality of the input data.
#         rank: Current process rank (for distributed training).
#         world_size: Total number of processes (for distributed training).
#         partition: Method to partition the data among processes (default: 'random')
#                     random: Randomly partition the data.
#                     sequential: Sequentially partition the data.
#     Returns:
#         local_x: Generated local input data.
#         local_y: Generated local output data.
#     """
#     if input_dim <= 0:
#         raise ValueError("Input dimension must be greater than 0.")
#     if num_samples <= 0:
#         raise ValueError("Number of samples must be greater than 0.")
#     if world_size <= 0:
#         raise ValueError("World size must be greater than 0.")
#     if rank < 0 or rank >= world_size:
#         raise ValueError("Rank must be between 0 and world_size - 1.")
    
    
#     # dimensions
#     if input_dim == 1:
#         # f(x) = 5 x^2 sin(12x) + (x^3 − 0.5) sin(3x − 0.5)+ 4 cos(2x) + noise,
#         if rank == 0:
#             print("Generating synthetic 1D training data...")
        
#         train_x = torch.linspace(0, 1, num_samples)
#         train_y = 5 * train_x**2 * torch.sin(12*train_x) + (train_x**3 - 0.5) * torch.sin(3*train_x - 0.5) + 4 * torch.cos(2*train_x) + torch.randn(train_x.size()) * 0.2


#     elif input_dim == 2:
#         # f(X) = (1 + ((x1 + x2 + 1)**2) * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) *
#         #         (30 + ((2*x1 - 3*x2)**2) * (18 - 32*x1 + 12*x1**2 + 12*x2 - 36*x1*x2 + 27*x2**2))
#         if rank == 0:
#             print("Generating Goldstein-Price 2D training data...")

#         train_x = torch.rand(num_samples, 2)  # Randomly generate 2D points in [0, 1]^2
        
#         train_y = (100 * (train_x[:, 1] - train_x[:, 0]**2)**2 + (1 - train_x[:, 0])**2) 
#         # train_y = (1 + ((train_x[:, 0] + train_x[:, 1] + 1)**2) * (19 - 14 * train_x[:, 0] + 3 * train_x[:, 0]**2 - 14 * train_x[:, 1] + 6 * train_x[:, 0] * train_x[:, 1] + 3 * train_x[:, 1]**2)) * \
#                 #  (30 + ((2 * train_x[:, 0] - 3 * train_x[:, 1])**2) * (18 - 32 * train_x[:, 0] + 12 * train_x[:, 0]**2 + 12 * train_x[:, 1] - 36 * train_x[:, 0] * train_x[:, 1] + 27 * train_x[:, 1]**2))
        
#         train_y += torch.randn(train_x.size(0)) * 0.2  # Add noise to the output
        

#     else:
#         raise ValueError("Input dimension must be either 1 or 2 for this function.")
    
    
#     if rank == 0:
#         print("global train_x shape:", train_x.shape)
#         print("global train_y shape:", train_y.shape)


#     # partition criteria
#     if partition == 'random':
#         # divide dataset into m parts based on world size and rank
#         # torch.manual_seed(42) 
#         local_indices = torch.randperm(train_x.size(0))
#         split_indices = torch.chunk(local_indices, world_size)
#         local_indices = split_indices[rank]
#         local_x = train_x[local_indices]
#         local_y = train_y[local_indices]

#     elif partition == 'sequential':
#         # divide dataset into m parts based on world size and rank
#         local_size = num_samples // world_size
#         start_idx = rank * local_size
#         end_idx = start_idx + local_size if rank < world_size - 1 else num_samples
#         local_x = train_x[start_idx:end_idx]
#         local_y = train_y[start_idx:end_idx]

#     elif partition == 'full':
#         # No partitioning, return the full dataset
#         local_x = train_x
#         local_y = train_y

#     else:
#         raise ValueError("Invalid partition method. Use 'random' or 'sequential'.")


#     if rank == 0:
#         print("local_x shape:", local_x.shape)
#         print("local_y shape:", local_y.shape)

#     return local_x, local_y


# def generate_test_data(num_samples, input_dim: int=1, rank: int=0, world_size: int=1,
#                        partition: str='random'):
#     """
#     Generate synthetic test data for Gaussian Processes.
#     Args:
#         num_samples: Number of samples to generate.
#         input_dim: Dimensionality of the input data.
#         rank: Current process rank (for distributed training).
#         world_size: Total number of processes (for distributed training).
#         partition: Method to partition the data among processes (default: 'random')
#                     random: Randomly partition the data.
#                     sequential: Sequentially partition the data.
#     Returns:
#         local_x: Generated local input data.
#     """
#     if input_dim <= 0:
#         raise ValueError("Input dimension must be greater than 0.")
#     if num_samples <= 0:
#         raise ValueError("Number of samples must be greater than 0.")
#     if world_size <= 0:
#         raise ValueError("World size must be greater than 0.")
#     if rank < 0 or rank >= world_size:
#         raise ValueError("Rank must be between 0 and world_size - 1.")
    
#     # dimensions
#     if input_dim == 1:
#         test_x = torch.linspace(0, 1, num_samples)
        
#     elif input_dim == 2:
#         test_x = torch.rand(num_samples, 2)  
#         # x1 = torch.linspace(-1, 1, num_samples)
#         # x2 = torch.linspace(-1, 1, num_samples)
#         # x1, x2 = torch.meshgrid(x1, x2)
#         # test_x = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    
    
#     else:
#         raise ValueError("Input dimension must be either 1 or 2 for this function.")

    
#     return test_x
    
#     # # partition criteria
#     # if partition == 'random':
#     #     # divide dataset into m parts based on world size and rank
#     #     torch.manual_seed(42) 
#     #     local_indices = torch.randperm(test_x.size(0))
#     #     split_indices = torch.chunk(local_indices, world_size)
#     #     local_indices = split_indices[rank]
#     #     local_x = test_x[local_indices]
#     #     local_y = test_y[local_indices]
        

#     # elif partition == 'sequential':
#     #     # divide dataset into m parts based on world size and rank
#     #     local_size = num_samples // world_size
#     #     start_idx = rank * local_size
#     #     end_idx = start_idx + local_size if rank < world_size - 1 else num_samples
#     #     local_x = test_x[start_idx:end_idx]
#     #     local_y = test_y[start_idx:end_idx]

#     # elif partition == 'full':
#     #     # No partitioning, return the full dataset
#     #     local_x = test_x
#     #     local_y = test_y

#     # else:
#     #     raise ValueError("Invalid partition method. Use 'random' or 'sequential'.")
    
#     # if rank == 0:
#     #     print("local_x shape:", local_x.shape)
#     #     print("local_y shape:", local_y.shape)

#     # return local_x, local_y


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