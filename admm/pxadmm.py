# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

import torch.multiprocessing as mp
import torch.distributed as dist

all__ = ["pxADMM", "pxadmm", "px_admm"]

class pxADMM(Optimizer):
    def __init__(self, params, rho: float=1.0, lip: float=1.0, rank: Optional[int]=0,
                        world_size: Optional[int]=1, tol_abs: float=1e-6, tol_rel: float=1e-4):
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
            lip: Lipschitz constant for gradient step size.
            rank: Process rank (default: 0).
            world_size: Total number of processes (default: 1).
            tol_abs: Absolute tolerance for residuals.
            tol_rel: Relative tolerance for residuals.

        No learning rate is used in pxADMM as no need of SGD
        """

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized. Please initialize it before using pxADMM.")
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available. Please install torch with distributed support.")
        
        print(f"pxADMM optimizer initialized with rho: {rho}, lip: {lip}, tol_abs: {tol_abs}, tol_rel: {tol_rel}")
        
        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if lip <= 0.0:
            raise ValueError("Lipschitz constant must be positive and greater than 0.0")
        if tol_abs <= 0.0:
            raise ValueError("Absolute tolerance must be positive and greater than 0.0")
        if tol_rel <= 0.0:
            raise ValueError("Relative tolerance must be positive and greater than 0.0")
        
        defaults = dict(rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel)
        super(pxADMM, self).__init__(params, defaults)

        self.rank = rank
        self.world_size = world_size
        self.iter = 0
        self.isConverged = False

        # flatten the parameters into a single vector
        self.param_shapes = []
        self.param_sizes = []
        flat_param = [] 

        for group in self.param_groups:
            for param in group['params']:
                self.param_shapes.append(param.shape)
                self.param_sizes.append(param.numel())
                flat_param.append(param.flatten())

        # list of flattened x parameters
        self.flat_param = torch.cat(flat_param).to(torch.float32) # flatten the parameters into a single vector
        self.total_params = self.flat_param.numel() # total number of parameters
        
        # Initialize z and lambda as single vectors
        self.state['flat']['z'] = self.flat_param.clone().detach().requires_grad_(False)
        self.state['flat']['lambda'] = torch.zeros_like(self.flat_param, requires_grad=False)
        
        # # initialize variables
        # for group in self.param_groups:
        #     for param in group['params']:
        #         # Initialize auxilary variable z and dual variable lambda
        #         self.state[param]['z'] = param.clone().detach().requires_grad_(False)
        #         self.state[param]['lambda'] = torch.zeros_like(param, requires_grad=False)

    
    def _unflatten_params(self, flat_param: Tensor) -> None:
        """
        Unflatten the parameters into their original shapes.
        Args:
            flat_param: Flattened parameter vector.
        """
        offset = 0
        i = 0
        for group in self.param_groups:
            for param in group['params']:
                size = self.param_sizes[i]
                param.data.copy_(flat_param[offset:offset + size].view(self.param_shapes[i]))
                offset += size
                i += 1


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
            tol_abs = group['tol_abs']
            tol_rel = group['tol_rel']

            # rather than iterating over the each parameter, we compute over the flattened parameters vector
            # Get state
            z_old = self.state['flat']['z']
            lambda_ = self.state['flat']['lambda']

            # Step 1: update auxiliary variable (z) using consensus same as cADMM
            # z^{k+1} = 1/M * sum_i (x_i^{k} + lambda_i^{k}/rho)

            # local update : (x_i^{k} + lambda_i^{k}/rho)
            z_new = self.flat_param.detach() + self.state['flat']['lambda'] / rho

            # synchronize z across all processes
            dist.all_reduce(z_new, op=dist.ReduceOp.SUM)
            z_new /= self.world_size 

            # Step 2: update the primal variable (x) using the proximal linearized gradient
            # x_i^{k+1} = z_i^k - (1/rho+ lip) * (\nabla L_i(z^k) + lambda_i^k)
            self._unflatten_params(z_new) #  z^{k+1} for gradient evaluation
            loss, grad = closure()

            # compute x_i^{k+1} 
            x_new = z_new - (1.0 / (rho + lip)) * (grad + lambda_)

            # Step 3: lambda-update
            # lambda_i^{k+1} = lambda_i^k + rho * (x_i^{k+1} - z^{k+1})
            lambda_new = lambda_ + rho * (x_new - z_new)

            # Update state
            self.state['flat']['z'] = z_new
            self.state['flat']['lambda'] = lambda_new
            
            self.flat_param.copy_(x_new)
            self._unflatten_params(x_new)

            # check convergence
            primal_residual = torch.norm(x_new - z_new, p=2)
            dual_residual = torch.norm(rho * (z_new - z_old), p=2)

            # adaptive tolerance 
            # will work in scaled version of pxADMM not for normal version
            # p = self.total_params
            # eps_primal = torch.sqrt(torch.tensor(p, dtype=torch.float)) * tol_abs + tol_rel * torch.max(torch.norm(x_new), torch.norm(z_new))
            # eps_dual = torch.sqrt(torch.tensor(p, dtype=torch.float)) * tol_abs + tol_rel * torch.norm(lambda_new)

            # # synchronize the residuals across all processes
            # dist.all_reduce(primal_residual, op=dist.ReduceOp.MAX)
            # dist.all_reduce(dual_residual, op=dist.ReduceOp.MAX)

            # # if primal_residual.item() < eps_primal and dual_residual.item() < eps_dual:
            # if primal_residual.item() < tol_abs and dual_residual.item() < tol_abs:
            #     self.isConverged = True
            #     if self.rank == 0:
            #         print("pxADMM converged at iteration {}".format(self.iter))
            #     return True


            constraint_norm = torch.norm(x_new - z_new, p=2)
            dist.all_reduce(constraint_norm, op=dist.ReduceOp.MAX)
            if constraint_norm.item() < tol_abs:
                self.isConverged = True
                if self.rank == 0:
                    print("pxADMM converged at iteration {}".format(self.iter))
                return True
            
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

