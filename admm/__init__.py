from .admm import ADMM, admm
from .cadmm import cADMM, cadmm, c_admm
from .pxadmm import pxADMM, pxadmm, px_admm
from .dec_admm import decADMM, decadmm, dec_admm

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

def pxadmm(params, **kwargs):
    """
    Create a pxADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the pxADMM optimizer.
        
    Returns:
        An instance of the pxADMM optimizer.
    """
    return pxADMM(params, **kwargs)

def dec_admm(params, **kwargs):
    """
    Create a decADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the decADMM optimizer.
        
    Returns:
        An instance of the decADMM optimizer.
    """
    return decADMM(params, **kwargs)

__all__ = ["ADMM", "admm"]
__all__ += ["cADMM", "cadmm", "c_admm"]
__all__ += ["pxADMM", "pxadmm", "px_admm"]
__all__ += ["decADMM", "decadmm", "dec_admm"]