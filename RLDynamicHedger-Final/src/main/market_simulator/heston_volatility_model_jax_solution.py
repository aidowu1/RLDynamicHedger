from jax import vmap
from jaxfin.price_engine.fft import (
    delta_call_fourier,
    fourier_inv_call,
)
from jaxfin.models.heston.heston import UnivHestonModel
import numpy as np
from typing import Tuple
from tqdm import tqdm

from src.main.market_simulator.heston_parameters import HestonParams
from src.main.market_simulator.simulator_results import HestonSimulationResults


class HestonOptionPricerWithJax:
    """
    Option pricing using Heston stochastic volatility using Paolo D’Elia JaxFin library
    Reference/repo can be found here:

    https://github.com/paolodelia99/jaxfin/tree/master

    It uses Monte Carlo simulation to compute:
        - Call price
        - Delta
    It uses the JaxFin library to do the computation which offers high performance vectorized computing
    via the Jax library
    JAX is a library for array-oriented numerical computation (à la NumPy), with
    automatic differentiation and JIT compilation to enable high-performance machine learning research.
    Further details on Jax can be found here:

    https://jax.readthedocs.io/en/latest/quickstart.html
    """

    def __init__(
            self,
            heston_params: HestonParams,
            S_0: float,
            V_0: float,
            K: float,
            r: float,
            n_paths: int,
            n_time_steps: int,
            time_to_expiry: float,
            seed: int,
    ):
        """
        Constructor
        :param heston_params: Heston parameters
        :param S_0: Initial asset price
        :param V_0: Initial volatility
        :param K: Strike price
        :param r: Risk free rate
        :param n_paths: Number of paths
        :param n_time_steps: Number of steps
        :param time_to_expiry: Time to expiry
        """
        self._S_0 = S_0
        self._V_0 = V_0
        self._K = K
        self._r = r
        self._n_paths = n_paths
        self._n_time_steps = n_time_steps
        self._time_to_expiry = time_to_expiry
        self._kappa = heston_params.kappa
        self._theta = heston_params.theta
        self._vol_of_vol = heston_params.sigma
        self._rho = heston_params.rho
        self._mean = heston_params.mean
        self._dt = self._time_to_expiry / self._n_time_steps
        self._seed = seed

    def _computePaths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the stochastic asset and volatility paths
        """
        heston_process = UnivHestonModel(
            s0=self._S_0,
            v0=self._V_0,
            mean=self._mean,
            kappa=self._kappa,
            theta=self._theta,
            sigma=self._vol_of_vol,
            rho=self._rho,
        )
        paths, variance_p = heston_process.sample_paths(
            self._time_to_expiry,
            self._n_time_steps,
            self._n_paths
        )
        self._asset_paths = np.asarray(paths)
        self._variance_process = np.asarray(variance_p).squeeze()
        return self._asset_paths, self._variance_process

    def _computeCallPricesAndDelta(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the call prices and deltas
        return: Call prices and deltas
        """
        prices = []
        deltas = []
        for j in tqdm(range(self._n_paths), desc="Simulation paths"):
            prices_per_path = [
                fourier_inv_call(
                    s0=self._asset_paths[i, j],
                    K=self._K,
                    T=self._time_to_expiry - i * self._dt,
                    v0=self._variance_process[i, j],
                    mu=self._r,
                    kappa=self._kappa,
                    theta=self._theta,
                    sigma=self._vol_of_vol,
                    rho=self._rho,
                )
                for i in range(self._n_time_steps)
            ]
            prices.append(prices_per_path)

            deltas_per_path = [
                delta_call_fourier(
                    self._asset_paths[i, j],
                    self._K,
                    self._time_to_expiry - i * self._dt,
                    self._variance_process[i, j],
                    self._r,
                    self._theta,
                    self._vol_of_vol,
                    self._kappa,
                    self._rho,
                )
                for i in range(self._n_time_steps)
            ]
            deltas.append(deltas_per_path)

        self._call_prices = np.asarray(prices)
        self._call_deltas = np.asarray(deltas)
        return self._call_prices, self._call_deltas

    def simulateHestonProcess(
            self,
    ) -> HestonSimulationResults:
        """
        Simulates the Monte Carlo Heston process to produce stochastic asset,volatility and option price paths
        :return: Returns asset price, volatility stochastic and option price paths
        :return:
        """
        self._computePaths()
        self._computeCallPricesAndDelta()
        result = HestonSimulationResults(
            stock_paths=self._asset_paths.T,
            volatility_paths=self._variance_process.T,
            option_price_paths=self._call_prices,
            option_deltas=self._call_deltas,
        )
        return result
