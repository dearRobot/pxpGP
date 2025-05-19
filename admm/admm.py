# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["ADMM", "admm"]

class ADMM(Optimizer):
    def __init__(self, params, rho: float=1.0,
                               max_iter: int=100,
                               lr: float=1e-3):
        """
        ADMM optimizer
        Args:
            params: Parameters to optimize.
            rho: Augmented Lagrangian parameter
            max_iter: Maximum number of iterations for ADMM
            lr: Learning rate for the primal variable (x) update
        """        
        
        print("ADMM optimizer initialized with rho: {}, max_iter: {}, lr: {}".format(rho, max_iter, lr))
        
        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive and greater than 0")
        if lr <= 0.0:
            raise ValueError("lr must be positive and greater than 0.0")
        
        defaults = dict(rho=rho, 
                        max_iter=max_iter,
                        lr=lr)
        super(ADMM, self).__init__(params, defaults)

        # initialize variables z, and dual variable lambda (non-scaled)
        print("lenght of param groups: ", len(self.param_groups))
        
        for group in self.param_groups:
            for param in group['params']:
                # if param.grad is None:
                #     continue

                # param is a x variable tensor so no need to initialize again
                # Initialize auxilary variable z
                self.state[param]['z'] = param.clone().detach().requires_grad_(False)
                # Initialize dual variable lambda
                self.state[param]['lambda'] = torch.zeros_like(param, requires_grad=False)


    def step(self, closure=None):
        """
        Performs one ADMM iteration.
        
        Args:
            closure: A callable that evaluates the loss f(x) and returns it.
                     Must return a tuple (loss, output) if used.
        """
        
        print("ADMM step called")

        if closure is None:
            raise ValueError("Closure must be provided computes the loss for ADMM step.")

        for group in self.param_groups:
            rho = group['rho']
            max_iter = group['max_iter']
            lr = group['lr']

            # Iterate over all parameters
            for param in group['params']:
                # if param.grad is None:
                #     continue

                # Get the state of the parameter
                z = self.state[param]['z']
                lambda_ = self.state[param]['lambda']

                # Step 1: update x i.e. the primal variable or model param
                # min f(x) + lambda^T (Ax - b) + rho/2 ||Ax - b||^2 (non scaled)
                param_old = param.clone().detach()
                optimizer = torch.optim.SGD([param], lr=0.01)

                for _ in range(max_iter):
                    optimizer.zero_grad()
                    # compute loss
                    loss, output = closure()
                    # loss.backward() # no need to call backward here
                    # ADMM penalty term
                    penalty = (lambda_ * (param.detach() - z)).sum() + (rho / 2) * torch.norm(param.detach() - z) ** 2
                    total_loss = loss + penalty

                    # compute gradients
                    total_loss.backward()

                    # Update the primal variable
                    optimizer.step()

                print("Loss: ", loss.item())


                # Step 2: update z i.e. the auxilary variable
                z_new = param.detach() + (lambda_ / rho)
        # soft thresholding operator  need to check
                z_new = torch.sign(z_new) * torch.clamp(torch.abs(z_new) - (1 / rho), min=0.0)

                # Step 3: update lambda i.e. the dual variable
                lambda_new = lambda_ + rho * (param.detach() - z_new)

                # update the state
                self.state[param]['z'] = z_new
                self.state[param]['lambda'] = lambda_new



def admm(params, **kwargs):
    """
    Create an ADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the ADMM optimizer.
        
    Returns:
        An instance of the ADMM optimizer.
    """
    return ADMM(params, **kwargs)


                

    

