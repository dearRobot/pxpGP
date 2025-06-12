# mypy: allow-untyped-defs
import torch
from torch import Tensor
from torch.optim import Optimizer

import torch.multiprocessing as mp
import torch.distributed as dist

__all__ = ["decpxADMM", "decpxadmm", "dec_pxadmm"]

class decpxADMM(Optimizer):
    def __init__(self, params, neighbors: list=None, rho: float=1.0, lip: float=1.0,
                 rank: int=0, world_size: int=1, tol_abs: float=1e-6, tol_rel: float=1e-4):
        """
        Decentralized Proximal Linearized ADMM (decpxADMM) optimizer for consensus optimization.
        Each process manages one agent's parameters, communicating only with neighbors.
        
        Args:
            params: Parameters to optimize.
            neighbors: List of neighbor process ranks for each parameter.
            rho: Augmented Lagrangian parameter.
            lip: Lipschitz constant for gradient step size.
            rank: Process rank (default: 0).
            world_size: Total number of processes (default: 1).
            tol_abs: Absolute tolerance for residuals.
            tol_rel: Relative tolerance for residuals.
        """
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized. Please initialize it before using decpxADMM.")
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
        
        if rank == 0:
            print(f"decpxADMM initialized with rho: {rho}, lip: {lip}, tol_abs: {tol_abs}, tol_rel: {tol_rel}, rank: {rank}, world_size: {world_size}")

        self.neighbors = neighbors
        self.degree = len(neighbors) if neighbors is not None else 0

        if self.degree < 1:
            raise ValueError("At least one neighbor must be specified for each parameter.")
        
        defaults = dict(rho=rho, lip=lip, tol_abs=tol_abs, tol_rel=tol_rel)
        super(decpxADMM, self).__init__(params, defaults)

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
        self.total_params = self.flat_param.numel()

        # Initialize alpha, x_prev as a single vector
        self.state['flat']['alpha'] = torch.zeros_like(self.flat_param, requires_grad=False)
        self.state['flat']['x_prev'] = self.flat_param.clone().detach().requires_grad_(False)


    def _unflatten_params(self, flat_param: Tensor) -> None:
        """
        Unflatten the parameters from a single vector to their original shapes.
        
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

    
    def get_neighbors_sum(self, x_prev: Tensor) -> Tensor:

        # receive my parameter x_i^k from neighbors
        recv_requests = []
        recv_x_prev = []

        for neighbor in self.neighbors:
            recv_x = torch.zeros_like(x_prev, device='cpu', requires_grad=False)
            recv_requests.append(dist.irecv(tensor=recv_x, src=neighbor))
            recv_x_prev.append(recv_x)

        # send my parameter x_i^k to neighbors
        send_requests = []
        x_prev_cpu = x_prev.cpu() if x_prev.is_cuda else x_prev

        for neighbor in self.neighbors:
            if not x_prev_cpu.is_contiguous():
                x_prev_cpu = x_prev_cpu.contiguous()
            try:
                send_requests.append(dist.isend(tensor=x_prev_cpu, dst=neighbor))
            except RuntimeError as e:
                print(f"Error sending data to neighbor {neighbor}: {e}")
        
        # wait for all receives to complete
        for req in recv_requests:
            req.wait()
        for req in send_requests:
            req.wait()

        # combine received parameters from neighbors
        neighbor_sum = torch.zeros_like(x_prev, device='cpu', requires_grad=False)
        for recv_x in recv_x_prev:
            neighbor_sum += recv_x

        neighbor_sum = neighbor_sum.to(x_prev.device)
        dist.barrier()
        return neighbor_sum


    def step(self, closure=None):
        """
        Performs one decpxADMM iteration.
        
        Args:
            closure: A callable that evaluates the loss f_i(x_i) and returns it.
        
        Returns:
            bool: True if convergence criteria are met, False otherwise.
        """

        if self.isConverged:
            return True
        
        if closure is None:
            raise ValueError("Closure must be provided to compute the loss for decpxADMM step.")
        
        self.iter += 1

        for group in self.param_groups:
            rho = group['rho']
            lip = group['lip']
            tol_abs = group['tol_abs']
            tol_rel = group['tol_rel']

            # rather than iterating over each parameter, we will use the flattened parameters
            alpha = self.state['flat']['alpha']
            x_prev = self.state['flat']['x_prev']

            # steps are in reverse order to avoid double communication and memory efficiency
            # Step 1: update alpha_i^k+1 (dual variable)
            # \alpha_i^{k+1} = \alpha_i^k + \rho (|N_i| x_i^{k+1} - \sum_{j \in N_i} x_j^{k+1}

            neighbor_sum = self.get_neighbors_sum(x_prev) 

            alpha_new = alpha + rho * (self.degree * x_prev - neighbor_sum)

            # Step 2: Update x_i^k+1 (primal variable)
            # x_i^{k+1} = (1 / lip + 2 * rho * self.degree) * (rho * \sum_{j \in N_i} x_j^{k} - 
            # \nabla f_i(x_i^{k}) - alpha_i^{k} + (rho * self.degree + lip) * x_i^{k})

            self._unflatten_params(x_prev) # x_prev is used for gradient evaluation
            loss, grad = closure()

            x_new = (1.0 / (lip + 2.0*rho*self.degree)) * (rho*neighbor_sum - grad - alpha_new +
                    (rho * self.degree + lip) * x_prev)

            # update state
            self.state['flat']['alpha'] = alpha_new
            self.state['flat']['x_prev'] = x_new
            self.flat_param.copy_(x_new)
            self._unflatten_params(x_new)

            # Check convergence criteria


def dec_pxadmm(params, **kwargs):
    """
    Create a decpxADMM optimizer instance.
    
    Args:
        params: Parameters to optimize.
        **kwargs: Additional arguments for decpxADMM.
    Returns:
        decpxADMM: Instance of the decpxADMM optimizer.
    """
    return decpxADMM(params, **kwargs)


def decpxadmm(params, **kwargs):
    """
    Alias for dec_pxadmm.
    
    Args:
        params: Parameters to optimize.
        **kwargs: Additional arguments for decpxADMM.
    Returns:
        decpxADMM: Instance of the decpxADMM optimizer.
    """
    return decpxADMM(params, **kwargs)