import numpy as np
import matplotlib.pyplot as plt

from src.main.market_simulator.simulator_results import HestonSimulationResults
from src.main.utility.enum_types import OptionType

np.random.seed(42)

class HestonModel:
    """
    Heston Stochastic Volatility Model class
    The heston model is defined by a system of SDEs, to describe the movement of asset prices,
    where an asset’s price and volatility follow random, Brownian motion processes
    (this is under real world measure)

    Also using this model we can compute the valuation of a European/American (Call/Put) option
    using Monte Carlo method
    """

    def __init__(
            self,
            s0: float,
            strike: float,
            r: float,
            q: float,
            expiry: float,
            kappa: float,
            theta: float,
            v0: float,
            vol_of_vol: float,
            rho: float,
            num_simulations: int,
            num_time_steps: int
    ):
        """
        Constructor
        :param s0: Initial stock price
        :param strike: Strike price
        :param r: Risk-free rate
        :param q: Dividend rate
        :param expiry: Maturity/expiry of option in years
        :param kappa: Rate of mean reversion of variance under risk-neutral dynamics
        :param v0: Initial volatility
        :param vol_of_vol: Volatility of volatility
        :param rho: Correlation between stock returns and variances under risk-neutral dynamics
        :param num_simulations: Number of time steps in simulation
        :param num_time_steps: Number of simulations
        """
        self._s0 = s0
        self._strike = strike
        self._r = r
        self._q = q
        self._expiry = expiry
        self._kappa = kappa
        self._theta = theta
        self._v0 = v0
        self._vol_of_vol = vol_of_vol
        self._rho = rho
        self._num_simulations = num_simulations
        self._num_time_steps = num_time_steps

        self._z1 = np.random.normal(size=(self._num_simulations, self._num_time_steps))
        self._z2 = self._rho * self._z1 + np.sqrt(1 - self._rho ** 2) * np.random.normal(
            size=(self._num_simulations, self._num_time_steps))

        self._dt = self._expiry / self._num_time_steps
        self._vt = np.zeros_like(self._z1)
        self._vt[:, 0] = self._v0
        self._st = np.zeros_like(self._z1)
        self._st[:, 0] = self._s0
        self._option_prices = np.zeros((self._num_simulations, self._num_time_steps))

    def simulateHestonProcess(
            self,
            option_type: OptionType = OptionType.call
    ) -> HestonSimulationResults:
        """
        Simulates the Monte Carlo Heston process to produce stochastic asset,volatility and option price paths
        :return: Returns asset price, volatility stochastic and option price paths
        """
        for i in range(1, self._num_time_steps):
            self._vt[:, i] = (self._vt[:, i - 1] + self._kappa * (self._theta - self._vt[:, i - 1]) * self._dt +
                              self._vol_of_vol * np.sqrt(np.maximum(0, self._vt[:, i - 1] * self._dt)) * self._z2[:, i])

            self._st[:, i] = (self._st[:, i - 1] * np.exp((self._r - self._q - 0.5 * self._vt[:, i]) * self._dt +
                                                          np.sqrt(np.maximum(0, self._vt[:, i] * self._dt)) * self._z1[
                                                                                                              :, i]))

            if option_type is OptionType.call:
                payoffs = np.maximum(self._st[:, i] - self._strike, 0)
            elif option_type is OptionType.put:
                payoffs = np.maximum(self._strike - self._st[:, i], 0)
            else:
                payoffs = np.maximum(self._st[:, i] - self._strike, 0)

            self._option_prices[:, i] = payoffs * np.exp(-self._r * (self._expiry - i * self._dt))

        result = HestonSimulationResults(
            stock_paths=self._st,
            volatility_paths=self._vt,
            option_price_paths=self._option_prices,
            option_deltas=None
        )
        return result

    @staticmethod
    def plotHestonModelPaths(
            simulation_results: HestonSimulationResults,
            expiry_in_years: int = 1,
    ):
        """
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 20))
        n_simulations, n_time_steps = simulation_results.stock_paths.shape
        time_steps = np.arange(0, expiry_in_years, expiry_in_years / n_time_steps)
        simulations = np.linspace(0, n_time_steps, n_simulations)
        ax1.plot(simulations, simulation_results.stock_paths)
        ax1.set_title('Heston Model Asset Prices')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Asset Prices')
        ax1.grid(True)

        ax2.plot(simulations, simulation_results.volatility_paths)
        ax2.set_title('Heston Model Variance Process')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Variance')
        ax2.grid(True)

        ax3.plot(simulations, simulation_results.option_price_paths)
        ax3.set_title('Heston Model Option Price Process')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Option Prices')
        ax3.grid(True)

        ax4.plot(time_steps, np.mean(simulation_results.option_price_paths, axis=0))
        ax4.set_title('Heston Model Call Option Price Monte Carlo Valuation')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Option Price')
        ax4.grid(True)

        plt.show()



