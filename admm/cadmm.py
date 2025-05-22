# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

import torch.multiprocessing as mp
import torch.distributed as dist

all__ = ["cADMM", "cadmm", "c_admm"]

class cADMM(Optimizer):
    def __init__(self, params, rho: float=1.0,
                                max_iter: int=100,
                                lr: float=1e-3, 
                                rank: Optional[int]=0,
                                world_size: Optional[int]=1):
        """
        cADMM optimizer
        Here we distribute dataset, parallelize computation and co-ordinate z-update among agents
        torch.multiprocessing is used to parallelize the computation
        Args:
            params: Parameters to optimize.
            rho: Augmented Lagrangian parameter
            max_iter: Maximum number of iterations for ADMM
            lr: Learning rate for the primal variable (x) update
            rank: Process rank (default: 0).
            world_size: Total number of processes (default: 1).
        """        
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized. Please initialize it before using cADMM.")
        
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available. Please install torch with distributed support.")
        
        print("cADMM optimizer initialized with rho: {}, max_iter: {}, lr: {}".format(rho, max_iter, lr))
        
        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive and greater than 0")
        if lr <= 0.0:
            raise ValueError("lr must be positive and greater than 0.0")
        
        defaults = dict(rho=rho, 
                        max_iter=max_iter,
                        lr=lr)
        super(cADMM, self).__init__(params, defaults)

        self.rank = rank    
        self.world_size = world_size

        # initialize variables
        for group in self.param_groups:
            for param in group['params']:
                # Initialize auxilary variable z and dual variable lambda
                self.state[param]['z'] = param.clone().detach().requires_grad_(False)
                self.state[param]['lambda'] = torch.zeros_like(param, requires_grad=False)


    def step(self, closure=None):
        """
        Performs one cADMM iteration.
        
        Args:
            closure: A callable that evaluates the loss f(x) and returns it.
                     Must return a tuple (loss, output) if used.
        """

        if closure is None:
            raise ValueError("Closure must be provided computes the loss for ADMM step.")
        
        for group in self.param_groups:            
            rho = group['rho']
            max_iter = group['max_iter']
            lr = group['lr']

            # Iterate over all parameters
            for param in group['params']:
                # Get the state of the parameter
                z = self.state[param]['z']
                lambda_ = self.state[param]['lambda']

                # Step 1: update x i.e. the primal variable or model param
                # param_old = param.clone().detach() # might be not needed
                optimizer_ = torch.optim.SGD([param], lr=lr)

                for _ in range(max_iter):
                    optimizer_.zero_grad()
                    loss, output = closure()
                    penalty = (lambda_ + rho * (param - z)).sum() + (rho / 2) * (param - z).pow(2).sum()
                    total_loss = loss + penalty
                    total_loss.backward()
                    optimizer_.step()

                # if self.rank == 0:
                    # print("Rank {}: loss: {}".format(self.rank, total_loss.item()))
                
                # Step 2: update z i.e. the auxilary variable
                # closed form solution for z in GP in general it is also an argmin problem
                # z^{k+1} = 1/M * sum_i (x_i^{k+1} + lambda_i^{k}/rho)
                
                # local update
                z_new = param.detach() + lambda_ / rho

                # synchronize z across all processes as a mean
                dist.all_reduce(z_new, op=dist.ReduceOp.SUM) # dont have AVG or MEAN directly available
                z_new /= self.world_size

                # soft thresholding
                z_new = torch.sign(z_new) * torch.clamp(torch.abs(z_new) - (1 / rho), min=0.0)

                # Step 3: update lambda i.e. the dual variable
                lambda_new = lambda_ + rho * (param.detach() - z_new)

                # update state
                self.state[param]['z'] = z_new
                self.state[param]['lambda'] = lambda_new
    
    
    # check if this is needed and actually affect the model
    def synchronize_parameters(self): 
        for group in self.param_groups:
            for param in group['params']:
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= self.world_size

def cadmm(params, **kwargs):
    """
     Create an cADMM optimizer
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for cADMM optimizer

    Returns:
        An instance of the ADMM optimizer.
    """
    return cADMM(params, **kwargs)

def c_admm(params, **kwargs):
    """
     Create an cADMM optimizer
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for cADMM optimizer

    Returns:
        An instance of the ADMM optimizer.
    """
    return cADMM(params, **kwargs)
