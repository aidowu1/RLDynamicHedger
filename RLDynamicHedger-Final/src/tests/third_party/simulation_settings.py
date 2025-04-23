


class SimulationSettings:
    """
    Simulation settings class
    """
    def __init__(self):
        """
        Constructor.
        """
        self.T = 60  # length of simulation 20= 1 month, 60 = 3 months
        self.S0 = 100  # starting price
        self.K = 100  # strike price
        self.sigma = 0.2  # volatility
        self.r = 0  # risk-free rate
        self.q = 0  # dividend yield
        self.mu = 0.05  # expected return on stock
        self.kappa = 0.01  # trading cost per unit traded
        self.dt = 1  # hedging time step
        self.notional = 100  # how many stocks the option is on
        self.rho = -0.4  # correlation of stochastic volatility process
        self.v = 0.6  # 0.04
        self.sigma0 = 0.2  # starting volatility
        self.c = 1.5  # standard deviation coefficient
        self.ds = 0.01

        self.n = 1000  # number of simulated paths
        self.days = 250  # number of days in a year
        self.freq = 1   # trading frequency was 3