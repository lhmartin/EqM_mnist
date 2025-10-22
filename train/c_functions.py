import torch


class LinearC():
    """
    Set C to 1-gamma for all gamma.
    """
    def __call__(self, gamma):
        return 1-gamma

class TruncatedDecayC():
    """
    Set C to 1 for gamma <= alpha, and decay linearly to 0 as gamma increases after alpha.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, gamma : torch.Tensor):
    
        mask = gamma <= self.alpha
        mask.to(gamma)
        
        c = (1- gamma) / (1 - self.alpha)
        c[mask] = 1.0
        
        return c

class PiecewiseC():
    """
    Set C to (beta - (beta - 1) * gamma / alpha) for gamma <= alpha, and 1 for gamma > alpha.
    Args:
        alpha: the alpha parameter
        beta: the beta parameter
    Returns:
        c: the c function
    """
    
    def __init__(self, alpha : float, beta : float):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, gamma : torch.Tensor):
        mask = gamma <= self.alpha
        
        c = (1 - gamma) / (1 - self.alpha)
        c[mask] = (self.beta - (self.beta - 1) * gamma[mask] / self.alpha)
        
        return c