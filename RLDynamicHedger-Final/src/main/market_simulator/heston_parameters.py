from dataclasses import dataclass

@dataclass
class HestonParams:
    """
    Heston parameters
    :param kappa:
    """
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    mean: float