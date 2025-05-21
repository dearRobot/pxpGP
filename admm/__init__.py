from .admm import ADMM, admm
from .cadmm import cADMM, cadmm, c_admm

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

def cadmm(params, **kwargs):
    """
    Create a cADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the cADMM optimizer.
        
    Returns:
        An instance of the cADMM optimizer.
    """
    return cADMM(params, **kwargs)

__all__ = ["ADMM", "admm"]
__all__ += ["cADMM", "cadmm", "c_admm"]