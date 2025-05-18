from .admm import ADMM, admm

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

__all__ = ["ADMM", "admm"]