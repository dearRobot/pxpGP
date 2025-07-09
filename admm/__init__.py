from .admm import ADMM, admm
from .cadmm import cADMM, cadmm, c_admm
from .pxadmm import pxADMM, pxadmm, px_admm
from .s_pxadmm import ScaledPxADMM, scaled_pxadmm, scaled_pxadmm
from .dec_admm import decADMM, decadmm, dec_admm
from .dec_pxadmm import decpxADMM, decpxadmm, dec_pxadmm
from .dec_s_pxadmm import DecScaledPxAdmm, dec_scaled_pxadmm, dec_scaled_px_admm

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

def scaled_pxadmm(params, **kwargs):
    """
    Create a scaled pxADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the scaled pxADMM optimizer.
        
    Returns:
        An instance of the scaled pxADMM optimizer.
    """
    return ScaledPxADMM(params, **kwargs)

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

def dec_pxadmm(params, **kwargs):
    """
    Create a decpxADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the decpxADMM optimizer.
        
    Returns:
        An instance of the decpxADMM optimizer.
    """
    return decpxADMM(params, **kwargs)

def dec_scaled_pxadmm(params, **kwargs):
    """
    Create a decScaledPxADMM optimizer.
    
    Args:
        params: Parameters to optimize.
        kwargs: Additional arguments for the decScaledPxADMM optimizer.
        
    Returns:
        An instance of the decScaledPxADMM optimizer.
    """
    return DecScaledPxAdmm(params, **kwargs)


__all__ = ["ADMM", "admm"]
__all__ += ["cADMM", "cadmm", "c_admm"]
__all__ += ["pxADMM", "pxadmm", "px_admm"]
__all__ += ["ScaledPxADMM", "scaled_pxadmm", "scaled_px_admm"]
__all__ += ["decADMM", "decadmm", "dec_admm"]
__all__ += ["decpxADMM", "decpxadmm", "dec_pxadmm"]
__all__ += ["DecScaledPxAdmm", "dec_scaled_pxadmm", "dec_scaled_px_admm"]