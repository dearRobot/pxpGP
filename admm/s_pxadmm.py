# mypy: allow-untyped-defs
import torch
from torch import Tensor
from torch.optim import Optimizer
import math

import torch.multiprocessing as mp
import torch.distributed as dist

__all__ = ["ScaledPxADMM", "scaled_pxadmm", "scaled_px_admm"]

class ScaledPxADMM(Optimizer):
    def __init__(self, params, rho: float=1.0, lip: float=1.0, rank: int=0,
                        world_size: int=1, tol_abs: float=1e-4, tol_rel: float=1e-2, dual: bool=True):
        """
        scaled pxADMM optimizer
        Here we distribute dataset, parallelize computation and co-ordinate z-update among agents
        torch.distributed is used to parallelize the computation

        scaled pxADMM handle distributed optimization especially for non-convex problems
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

        No learning rate is used in scaled pxADMM as no need of SGD
        """

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized. Please initialize it before using pxADMM.")
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available. Please install torch with distributed support.")
                
        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if lip <= 0.0:
            raise ValueError("Lipschitz constant must be positive and greater than 0.0")
        if tol_abs <= 0.0:
            raise ValueError("Absolute tolerance must be positive and greater than 0.0")
        if tol_rel <= 0.0:
            raise ValueError("Relative tolerance must be positive and greater than 0.0")
        
        defaults = dict(rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel)
        super(ScaledPxADMM, self).__init__(params, defaults)

        self.rank = rank
        self.world_size = world_size
        self.iter = 0
        self.isConverged = False
        self.dual = dual  

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
        self.state['flat']['u'] = torch.zeros_like(self.flat_param, requires_grad=False)
        self.state['flat']['old_grad'] = torch.zeros_like(self.flat_param, requires_grad=False)

        self.state['flat']['m'] = torch.zeros_like(self.flat_param, requires_grad=False)  # Momentum
        
    
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


    # TODO: Implement scaled pxADMM and adaptive tolerance
    def step(self, closure=None, epoch: int=1):
        """
        Performs a single optimization step.
        This step includes:
        - updating local primal variables using proximal inexact (linearized) gradient,
        - computing consensus over z using all-reduce (if enabled),
        - updating dual variables (lambda),
        - update rho and lip parameters 
        - checking for convergence based on residuals.

        Args:
            closure (callable): A closure that reevaluates the model and returns (loss, gradient).

        Returns:
            bool: True if converged based on stopping condition; otherwise, False.
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

            z_old = self.state['flat']['z']
            u_ = self.state['flat']['u']
            grad_old = self.state['flat']['old_grad']

            m = self.state['flat']['m']  # Momentum buffer
                        
            # Step 1: update auxiliary variable (z) using consensus same as cADMM // z^{k+1} = 1/M * sum_i (x_i^{k} + u_i^{k})
            z_new = self.flat_param.detach() + u_

            dist.all_reduce(z_new, op=dist.ReduceOp.SUM) # synchronize z
            z_new /= self.world_size 

            v_ = z_new - u_  # v = z^{k+1} - u_i^{k}
            
            # Step 2: update the primal variable (x) using the proximal linearized gradient
            # x_i^{k+1} = z_i^k - (1/rho+ lip) * (\nabla L_i(z^k) + rho * u_i^k)
            self._unflatten_params(v_)
            loss, grad = closure()
            f_old = loss.item()

            # adaptive momentum from Adam optimizer
            # beta1 = 0.9 
            # m.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            alpha = 1.0 / (rho + lip)

            # x_try = v_ - alpha * m  
            x_try = v_ - alpha * grad
            self._unflatten_params(x_try)
            f_try = closure()[0].item()

            # Armijo constants
            c, tau = 1e-4, 0.5
            iter = 0

            # while f_try > f_old + 0.1 * lip * torch.norm(x_try - v)**2 and iter < 10: # Armijo condition
            while (f_try > f_old - c*alpha*torch.dot(grad.flatten(), grad.flatten())) and (iter < 10):
                lip *= tau
                alpha = 1.0 / (rho + lip)
                # x_try = v_ - alpha * m 
                x_try = v_ - alpha * grad
                self._unflatten_params(x_try)
                f_try = closure()[0].item()
                iter += 1
            
            x_new = x_try

            # noise addition
            # noise_temp = 0.01  # Initial noise temperature
            # noise_decay = 0.01  # Decay factor for noise temperature

            # T = noise_temp / (1.0 + noise_decay * self.iter)
            # sigma = T * 0.05  # Scale noise by temperature
            # noise = torch.randn_like(x_new) * sigma
            # x_new += noise
            # sigma = 0.02 / math.sqrt(self.iter + 1)
            # noise = torch.randn_like(x_new) * sigma
            # x_new += noise
            
            # Step 3: u-update // u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}
            primal_residual = x_new - z_new
            u_new = u_ + primal_residual
            
            self.flat_param.copy_(x_new)
            self._unflatten_params(x_new)

            # check convergence
            r_norm = torch.norm(primal_residual, p=2)
            s_norm = torch.norm(rho * (z_new - z_old), p=2)
                        
            # adaptive tolerance 
            p = self.total_params
            eps_primal = torch.sqrt(torch.tensor(p, dtype=torch.float)) * tol_abs + tol_rel * torch.max(torch.norm(x_new), torch.norm(z_new))
            eps_dual = torch.sqrt(torch.tensor(p, dtype=torch.float)) * tol_abs + tol_rel * torch.norm(rho * u_new)

            dist.all_reduce(eps_primal, op=dist.ReduceOp.SUM) #op=dist.ReduceOp.MAX)
            dist.all_reduce(eps_dual, op=dist.ReduceOp.SUM) #op=dist.ReduceOp.MAX)
            dist.all_reduce(r_norm, op=dist.ReduceOp.SUM) #op=dist.ReduceOp.MAX)
            dist.all_reduce(s_norm, op=dist.ReduceOp.SUM) #op=dist.ReduceOp.MAX)
            eps_primal /= self.world_size
            eps_dual /= self.world_size
            r_norm /= self.world_size
            s_norm /= self.world_size

            if self.rank == 0 and self.iter % 10 == 0:
                print(f'rank {self.rank}, epoch {epoch}, loss: {loss.item()}, rho: {rho:.4f}, lip: {lip:.4f}')

            if self.dual:
                if r_norm.item() < eps_primal and s_norm.item() < eps_dual:
                    self.isConverged = True
                    if self.rank == 0:
                        print("scaled pxADMM converged at iteration {}".format(self.iter))
                    return True
            else:
                if r_norm.item() < eps_primal:
                    self.isConverged = True
                    if self.rank == 0:
                        print("scaled pxADMM converged at iteration {}".format(self.iter))
                    return True

            # update rho
            if r_norm.item() > 10 * s_norm.item():
                rho *= 2.0
            elif s_norm.item() > 10 * r_norm.item():
                rho /= 2.0

            rho = max(1.0e-3, min(100.0, rho))

            # update lip
            beta = 0.9
            diff_norm = torch.norm(grad - grad_old) / (torch.norm(z_new - z_old) + 1e-8)
            lip_new = beta * lip + (1 - beta) * diff_norm.item()
            lip_new = max(1.0e-3, min(1000, lip_new))
            
            # Update state
            self.state['flat']['z'] = z_new
            self.state['flat']['u'] = u_new
            self.state['flat']['old_grad'] = grad.clone().detach()

            group['rho'] = rho
            group['lip'] = lip_new
 
        return False    
    

    def synchronize_parameters(self):  # not used for now
        for group in self.param_groups:
            for param in group['params']:
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= self.world_size


def scaled_pxadmm(params, **kwargs):
    """
    Create a pxADMM optimizer.
    Args:
        params (iterable): Iterable of model parameters to optimize.
        **kwargs: Additional arguments passed to the pxADMM constructor, such as:
            - rho (float): Augmented Lagrangian parameter.
            - lip (float): Lipschitz constant.
            - rank (int): Process rank.
            - world_size (int): Total number of distributed processes.

    Returns:
        pxADMM: An instance of the pxADMM optimizer.
    """
    return ScaledPxADMM(params, **kwargs)


def scaled_px_admm(params, **kwargs):
    """
    Create a pxADMM optimizer.
    Args:
        params (iterable): Iterable of model parameters to optimize.
        **kwargs: Additional arguments passed to the pxADMM constructor, such as:
            - rho (float): Augmented Lagrangian parameter.
            - lip (float): Lipschitz constant.
            - rank (int): Process rank.
            - world_size (int): Total number of distributed processes.

    Returns:
        pxADMM: An instance of the pxADMM optimizer.
    """
    return ScaledPxADMM(params, **kwargs)

