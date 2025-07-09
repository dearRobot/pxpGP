import torch
from torch import Tensor
from torch.optim import Optimizer

import torch.multiprocessing as mp
import torch.distributed as dist

__all__ = ["decADMM", "decadmm", "dec_admm"]

class decADMM(Optimizer):
    def __init__(self, params, neighbors: list=None, rho: float=1.0, max_iter: int=100,
                                lr: float=1e-3, rank: int=0, world_size: int=1):
                   
        """
        Decentralized ADMM (decADMM) optimizer for consensus optimization.
        Each process manages one agent's parameters, communicating only with neighbors.
        Implements updates from Equation (9) of Shi et al., "On the Linear Convergence of 
        the ADMM in Decentralized Consensus Optimization".
        
        Args:
            params: Parameters to optimize.
            neighbors: List of neighbor process ranks for each parameter.

            rho: Augmented Lagrangian parameter
            max_iter: Maximum number of iterations for ADMM
            lr: Learning rate for the primal variable (x) update
            rank: Process rank (default: 0).
            world_size: Total number of processes (default: 1).
        """        
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized. Please initialize it before using decADMM.")
        
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available. Please install torch with distributed support.")
                
        if rho <= 0.0:
            raise ValueError("rho must be positive and greater than 0.0")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive and greater than 0")
        if lr <= 0.0:
            raise ValueError("lr must be positive and greater than 0.0")
        
        if rank == 0:
            print(f"decADMM initialized with rho: {rho}, max_iter: {max_iter}, lr: {lr}, rank: {rank}, world_size: {world_size}")
        
        self.neighbors = neighbors
        self.degree = len(neighbors) if neighbors is not None else 0

        if self.degree < 1:
            raise ValueError("At least one neighbor must be specified for each parameter.")
        
        
        defaults = dict(rho=rho, max_iter=max_iter, lr=lr)
        super(decADMM, self).__init__(params, defaults)

        self.rank = rank    
        self.world_size = world_size

        # initialize variables
        for group in self.param_groups:
            for param in group['params']:
                # Initialize dual variable alpha and x_i^k for L+
                self.state[param]['alpha'] = torch.zeros_like(param, requires_grad=False)
                self.state[param]['x_prev'] = param.clone().detach().requires_grad_(False)

        
    def step(self, closure=None):
        """
        Performs one decADMM iteration.
        
        Args:
            closure: A callable that evaluates the loss f_i(x_i) and returns it.
                        Must return a tuple (loss, output) if used.
        
        \nabla f_i(x_i) + \alpha_i^k + 2c |N_i| x_i - c (|N_i| x_i^k + \sum_{j \in N_i} x_j^k) = 0 
        
        """

        if closure is None:
            raise ValueError("Closure must be provided computes the loss for decADMM step.")

        for group in self.param_groups:            
            rho = group['rho']
            max_iter = group['max_iter']
            lr = group['lr']

            for param in group['params']:
                alpha = self.state[param]['alpha']
                x_prev = self.state[param]['x_prev']

                # Step 1: Update x_i^k+1 (primal variable)
                
                # receive my parameter x_i^k from neighbors
                recv_request = []
                recv_x_prev = []

                for neighbor_rank in self.neighbors:
                    recv_x = torch.zeros_like(param, device='cpu', requires_grad=False)
                    recv_request.append(dist.irecv(tensor=recv_x, src=neighbor_rank))
                    recv_x_prev.append(recv_x)
                
                # send my parameter x_i^k to neighbors
                send_rquest = []
                x_prev_cpu = x_prev.cpu() if x_prev.is_cuda else x_prev

                for neighbor_rank in self.neighbors:
                    if not x_prev_cpu.is_contiguous():
                        x_prev_cpu = x_prev_cpu.contiguous()
                    try:
                        send_rquest.append(dist.isend(tensor=x_prev_cpu, dst=neighbor_rank))
                    except RuntimeError as e:
                        print(f"Rank {self.rank}: isend to {neighbor_rank} failed: {e}")
                        raise

                # Wait for all receives to complete
                for req in recv_request:
                    req.wait()
                
                for req in send_rquest:
                    req.wait()
                
                # Combine received parameters from neighbors
                neighbor_sum_prev = torch.zeros_like(param, device='cpu', requires_grad=False)
                for recv_x in recv_x_prev:
                    neighbor_sum_prev += recv_x
                
                neighbor_sum_prev = neighbor_sum_prev.to(param.device) 
                
                dist.barrier()  
   
                # x_i^k+1 update:
                # \nabla f_i(x_i) + \alpha_i^k + 2c |N_i| x_i - c (|N_i| x_i^k + \sum_{j \in N_i} x_j^k) = 0 
# TODO: Try with Adam optimizer
                
                optimizer = torch.optim.SGD([param], lr=lr)
                for _ in range(max_iter):
                    optimizer.zero_grad()
                    loss, _ = closure()
                    penalty = alpha + (2* rho * self.degree * param) - rho * (self.degree * x_prev + neighbor_sum_prev)
                    total_loss = loss + penalty.sum()
                    total_loss.backward()
                    optimizer.step()
                
                # Step 2: Update alpha_i^k+1 (dual variable)
                # \alpha_i^{k+1} = \alpha_i^k + \rho (|N_i| x_i^{k+1} - \sum_{j \in N_i} x_j^{k+1}

                # receive updated parameters x_j^{k+1} from neighbors
                recv_request = []
                recv_x_curr = []

                for neighbor_rank in self.neighbors:
                    recv_x = torch.zeros_like(param, device='cpu', requires_grad=False)
                    recv_request.append(dist.irecv(tensor=recv_x, src=neighbor_rank))
                    recv_x_curr.append(recv_x)

                # send updated parameter x_i^{k+1} to neighbors
                send_request = []
                param_cpu = param.clone().detach().cpu() if param.is_cuda else param.clone().detach()

                for neighbor_rank in self.neighbors:
                    if not param_cpu.is_contiguous():
                        param_cpu = param_cpu.contiguous()
                    try:
                        send_request.append(dist.isend(tensor=param_cpu, dst=neighbor_rank))
                    except RuntimeError as e:
                        print(f"Rank {self.rank}: isend to {neighbor_rank} failed: {e}")
                        raise

                # Wait for all receives to complete
                for req in recv_request:
                    req.wait()

                for req in send_request:
                    req.wait()
                
                # combine received parameters from neighbors
                neighbor_sum_curr = torch.zeros_like(param, device='cpu', requires_grad=False)
                for recv_x in recv_x_curr:
                    neighbor_sum_curr += recv_x

                neighbor_sum_curr = neighbor_sum_curr.to(param.device)

                dist.barrier()

                # update alpha_i^{k+1}
                alpha_new = alpha + rho * (self.degree * param.detach() - neighbor_sum_curr)

                # Update state
                self.state[param]['alpha'] = alpha_new
                self.state[param]['x_prev'] = param.clone().detach().requires_grad_(False) 

            
def decadmm(params, **kwargs):
    """
    Factory function to create a decADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        **kwargs: Additional arguments for the decADMM optimizer.
    
    Returns:
        decADMM optimizer instance.
    """
    return decADMM(params, **kwargs)

def dec_admm(param, **kwargs):
    """
    Alias for decadmm function.
    
    Args:
        param: Parameters to optimize.
        **kwargs: Additional arguments for the decADMM optimizer.
    
    Returns:
        decADMM optimizer instance.
    """
    return decADMM(param, **kwargs)
