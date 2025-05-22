# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

import torch.multiprocessing as mp
import torch.distributed as dist

all__ = ["pxADMM", "pxadmm", "px_admm"]

class pxADMM(Optimizer):
    def __init__(self, params, rho: float=1.0,
                                 max_iter: int=100,
                                 rank: Optional[int]=0,
                                 world_size: Optional[int]=1):
        """
        pxADMM optimizer
        Here we distribute dataset, parallelize computation and co-ordinate z-update among agents
        torch.distributed is used to parallelize the computation

        pxADMM handle distributed optimization especially for non-convex problems
        proximal linearized cADMM is used to solve the optimization problem
        In x update step, instead of using exact gradient minimization, we use proximal linearized gradient
            computated using first order Taylor expansion around current globla estimate z^k

        Args:
            params: Parameters to optimize.
            rho: Augmented Lagrangian parameter
            max_iter: Maximum number of iterations for ADMM
            rank: Process rank (default: 0).
            world_size: Total number of processes (default: 1).

        No learning rate is used in pxADMM as no need of SGD
        """

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized. Please initialize it before using pxADMM.")
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available. Please install torch with distributed support.")
        
        print("pxADMM optimizer initialized with rho: {}, max_iter: {}, lr: {}".format(rho, max_iter, lr))

        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive and greater than 0")
        
        defaults = dict(rho=rho, 
                        max_iter=max_iter,
                        lr=lr)
        super(pxADMM, self).__init__(params, defaults)

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
        Performs a single optimization step.
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            max_iter = group['max_iter']

            # Iterate over the parameters
            for param in group['params']:
                # get the state of the parameter
                z = self.state[param]['z']
                lambda_ = self.state[param]['lambda']

                # Step 1: update the primal variable (x) using the proximal linearized gradient
            

        

def pxadmm(params, **kwargs):
    """
    Create a pxADMM optimizer.
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for pxADMM.
    """
    return pxADMM(params, **kwargs)

def px_admm(params, **kwargs):
    """
    Create a pxADMM optimizer.
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for pxADMM.
    """
    return pxADMM(params, **kwargs)

