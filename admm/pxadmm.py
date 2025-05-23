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
                                lip: float=1.0,
                                tol: float=1e-6,
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
        
        print("pxADMM optimizer initialized with rho: {}, lip: {}, tol: {}".format(rho, lip, tol))

        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if lip <= 0.0:
            raise ValueError("Lipschitz constant must be positive and greater than 0.0")
        if tol <= 0.0:
            raise ValueError("Tolerance must be positive and greater than 0.0")
        
        defaults = dict(rho=rho, 
                        lip=lip, 
                        tol=tol)
        super(pxADMM, self).__init__(params, defaults)

        self.rank = rank
        self.world_size = world_size
        self.iter = 0
        self.isConverged = False
        

        # initialize variables
        for group in self.param_groups:
            for param in group['params']:
                # Initialize auxilary variable z and dual variable lambda
                self.state[param]['z'] = param.clone().detach().requires_grad_(False)
                self.state[param]['lambda'] = torch.zeros_like(param, requires_grad=False)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        lip : Lipschitz constant of the gradient of the loss function
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if self.isConverged:
            return True
        
        if closure is None:
            raise ValueError("Closure must be provided computes the loss and gradient for pxADMM step.")

        self.iter += 1
        
        for group in self.param_groups:
            rho = group['rho']
            lip = group['lip']
            tol = group['tol']

            
            # print group parameters
            # print("Rank {}: group parameters are {}".format(self.rank, [p.shape for p in group['params']]))
            
            # loss, grad = closure()
            
            # Iterate over the parameters
            for i, param in enumerate(group['params']):
                # get the state of the parameter
                z = self.state[param]['z']
                lambda_ = self.state[param]['lambda']

                # Step 1: update auxiliary variable (z) using consensus same as cADMM
                # z^{k+1} = 1/M * sum_i (x_i^{k+1} + lambda_i^{k}/rho)
                
                # local update
                z_new = param.detach() + lambda_ / rho

                # synchronize z across all processes
                dist.all_reduce(z_new, op=dist.ReduceOp.SUM)
                z_new /= self.world_size 

                # Step 2: update the primal variable (x) using the proximal linearized gradient
                # x_i^{k+1} = z_i^k - (1/rho+ lip) * (\nabla L_i(z^k) + lambda_i^k)
                param_old = param.clone().detach() 
                param.data.copy_(z_new) # to compute the gradient at z^k+1
                loss, grad = closure()

                # can get loss here and compute the gradient wrt z_new
                
                # print in normal cpu format
                # print("Rank {}: z_newm is {}".format(self.rank, z_new))
                # print("Rank {}: param_old is {}".format(self.rank, param_old))
                # print("Rank {}: grad is {}".format(self.rank, grad))
                # print("Rank {}: lambda is {}".format(self.rank, lambda_))

                # compute x_i^{k+1} 
                param.data.copy_(z_new - (1.0 / (rho + lip)) * (grad[i] + lambda_))

                # Step 3: update the dual variable (lambda)
                # lambda_i^{k+1} = lambda_i^k + rho * (x_i^{k+1} - z_i^{k+1})
                lambda_new = lambda_ + rho * (param.detach() - z_new)

                # update the parameter
                self.state[param]['z'] = z_new
                self.state[param]['lambda'] = lambda_new

        # check convergence

        return False    

    def synchronize_parameters(self):  # not used for now
        for group in self.param_groups:
            for param in group['params']:
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= self.world_size

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

