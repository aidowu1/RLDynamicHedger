import numpy as np
from collections import namedtuple
from scipy.stats import norm
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict

import src.main.configs_global as configs
from src.main.utility.settings_reader import SettingsReader
from src.main.market_simulator.parameters import Parameters
from src.main.market_simulator.simulator_results import (BlackScholesCallResults, SABRSimulationResults,
                                                         GBMSimulationResults, SABRSimulationRunResults,
                                                         HedgingStrategyResults, AdjustedPnlProcessResults,
                                                         ClassicalHedgingResults, HestonSimulationResults)
from src.main.utility.utils import Helpers
from src.main.market_simulator.heston_volatility_model_alternate_solution import HestonModel
from src.main.market_simulator.heston_volatility_model_jax_solution import HestonOptionPricerWithJax
from src.main.market_simulator.heston_parameters import HestonParams
from src.main.utility.logging import Logger

# Set the ramdom seed
np.random.seed(configs.RANDOM_SEED)

class MarketSimulator:
    """
    Market simulator component used to simulate a market.
    It simulates the paths of a stock and the price of a call option using the following
    models:
        - Geometric Brownian Motion (GBM)
        - Heston volatility model
        - Stochastic Alpha Beta Tho (SABR) model
    """
    def __init__(
            self,
            parameter_settings_filename: str = None,
            parameters: Parameters = None
    ):
        """
        Constructor
        :param parameter_settings_filename: Parameter settings filename
        :type parameter_settings_filename: str
        """
        if parameter_settings_filename is not None:
            parameter_settings_data = Helpers.getParameterSettings(parameter_settings_filename)
            self._parameters = Parameters(**parameter_settings_data)
        else:
            self._parameters = parameters
        times = np.arange(0, self._parameters.n_time_steps, self._parameters.trading_frequency)
        self._current_time_steps = times / self._parameters.n_days_per_year
        self._expiry_time = self._parameters.n_time_steps / self._parameters.n_days_per_year

        self._time_horizon = int(self._parameters.n_time_steps / self._parameters.trading_frequency)
        self._logger = Logger.getLogger()

    def computeBlackScholesCall(
            self,
            current_stock_price: np.ndarray,
            current_volatility: Union[np.ndarray, float]
    ) -> BlackScholesCallResults:
        """
        Calculates the Black Scholes price and delta for a call option.
        :param current_stock_price: Current stock price
        :param current_volatility: Current volatility
        :return: Price and delta for a call option
        """
        d1 = ((np.log(current_stock_price / self._parameters.strike_price)
               + (self._parameters.risk_free_rate - self._parameters.dividend_rate + 0.5
                  * current_volatility ** 2)
               * (self._expiry_time - self._current_time_steps))
              / (current_volatility * np.sqrt(self._expiry_time - self._current_time_steps)))
        d2 = d1 - (current_volatility * np.sqrt(self._expiry_time - self._current_time_steps))

        # price
        price = (current_stock_price * np.exp(-self._parameters.dividend_rate *
                                              (self._expiry_time - self._current_time_steps))
                 * norm.cdf(d1) - np.exp(-self._parameters.risk_free_rate *
                                         (self._expiry_time - self._current_time_steps))
                 * self._parameters.strike_price * norm.cdf(d2))

        # delta
        delta = np.exp(-self._parameters.dividend_rate * (self._expiry_time - self._current_time_steps)) * norm.cdf(d1)

        results = BlackScholesCallResults(
            price=price,
            delta=delta
        )

        return results

    def computeGBMSimulation(self) -> np.ndarray:
        """
        Computes the Geometric Brownian Motion (GBM) simulation of the underlying stock price
        :return: Underlying stock price
        """
        underlying_price = np.zeros((self._parameters.n_paths, self._time_horizon))

        underlying_price[:, 0] = self._parameters.start_stock_price

        # for t in tqdm(range(1, self._parameters.n_time_steps)):  # generate paths
        for t in tqdm(range(1, self._time_horizon)):  # generate paths
            dW = np.random.normal(0, 1, size=self._parameters.n_paths)
            underlying_price[:, t] = underlying_price[:, t - 1] * np.exp(
                (self._parameters.return_on_stock - 0.5 * self._parameters.volatility ** 2)
                * self._parameters.hedging_time_step
                + self._parameters.volatility * np.sqrt(self._parameters.hedging_time_step) * dW)

        return underlying_price

    def computeSABRSimulation(self) -> SABRSimulationResults:
        """
        Computes the Stochastic Alpha Beta Rho (SABR) simulation of the underlying stock price
        :return: Underlying stock price and the stochastic volatility
        """
        stochastic_volatility = np.zeros((self._parameters.n_paths, self._time_horizon))
        underlying_price = np.zeros((self._parameters.n_paths, self._time_horizon))

        stochastic_volatility[:, 0] = self._parameters.start_volatility
        underlying_price[:, 0] = self._parameters.start_stock_price

        # generate parameters for creating correlated random numbers
        mean = np.array([0, 0])
        corr = np.array([[1, self._parameters.rho], [self._parameters.rho, 1]])  # Correlation matrix
        std = np.diag([1, 1])  # standard deviation vector
        cov = std @ corr @ std  # covariance matrix, input of multivariate_normal function

        # generate price path based on random component and derive option price and delta
        for t in tqdm(range(1, self._time_horizon)):
            # correlated random BM increments
            dW = np.random.multivariate_normal(mean, cov, size=self._parameters.n_paths)
            stochastic_volatility[:, t] = stochastic_volatility[:, t - 1] * np.exp(
                (-0.5 * self._parameters.volatility_of_volatility ** 2) * self._parameters.hedging_time_step
                + self._parameters.volatility_of_volatility
                * np.sqrt(self._parameters.hedging_time_step) * dW[:, 0])
            underlying_price[:, t] = underlying_price[:, t - 1] * np.exp(
                (self._parameters.return_on_stock - 0.5 * stochastic_volatility[:, t] ** 2)
                * self._parameters.hedging_time_step
                + stochastic_volatility[:, t]
                * np.sqrt(self._parameters.hedging_time_step) * dW[:, 1])

        results = SABRSimulationResults(
            underlying_price=underlying_price,
            stochastic_volatility=stochastic_volatility
        )

        return results

    def computeImpliedVolatility(
            self,
            stock_price: np.ndarray,
            stochastic_volatility
    ) -> np.ndarray:
        """
        Computes the implied volatility
        :param stock_price: Underlying stock price
        :param stochastic_volatility: Stochastic volatility
        :return: Implied volatility
        """
        f = stock_price * np.exp((self._parameters.risk_free_rate - self._parameters.dividend_rate)
                                 * (self._expiry_time - self._current_time_steps))
        # at the money case
        atm = stochastic_volatility * (1 + (self._expiry_time - self._current_time_steps)
                                       * (self._parameters.rho * self._parameters.volatility_of_volatility * stochastic_volatility
                                       / 4 + self._parameters.volatility_of_volatility ** 2 * (2 - 3 * self._parameters.rho ** 2) / 24))
        xi = (self._parameters.volatility_of_volatility / stochastic_volatility) * np.log(f / self._parameters.strike_price)
        xi_func = np.log((np.sqrt(1 - 2 * self._parameters.rho * xi + xi ** 2)
                          + xi - self._parameters.rho) / (1 - self._parameters.rho))

        imp_vol = np.where(f == self._parameters.strike_price, atm, atm * xi / xi_func)

        return imp_vol

    def computeBartlettDelta(
            self,
            sabr_stock_price: np.ndarray,
            implied_volatility: np.ndarray
            ):
        """
        Computes the "Bartlett's" Delta using numerical differentiation
        Reference: # following Bartlett (2006) Eq. 12
        :param sabr_stock_price: Sabr stock price
        :param implied_volatility: Implied volatility
        :return:
        """

        d_volatility = (self._parameters.central_difference_spacing * self._parameters.volatility_of_volatility
                        * self._parameters.rho / sabr_stock_price)
        i_sigma = self.computeImpliedVolatility(sabr_stock_price, implied_volatility)
        i_sigma_plus = self.computeImpliedVolatility(
            sabr_stock_price + self._parameters.central_difference_spacing,
            implied_volatility + d_volatility)

        price_base_results = self.computeBlackScholesCall(
            current_stock_price=sabr_stock_price,
            current_volatility=i_sigma
        )
        price_plus_results = self.computeBlackScholesCall(
            current_stock_price=sabr_stock_price + self._parameters.central_difference_spacing,
            current_volatility=i_sigma_plus
        )

        # finite differences
        bartlett_delta = (price_plus_results.price - price_base_results.price) / self._parameters.central_difference_spacing

        return bartlett_delta

    def runGBMSimulation(self) -> GBMSimulationResults:
        """
        Runs the GBM simulation
        :return: Returns the stock price, call price and delta
        """
        gbm_stock_paths = self.computeGBMSimulation()
        gbm_black_scholes_results = self.computeBlackScholesCall(
            current_stock_price=gbm_stock_paths,
            current_volatility=self._parameters.volatility
        )

        gbm_results = GBMSimulationResults(
            gbm_stock_paths=gbm_stock_paths,
            gbm_call_price=gbm_black_scholes_results.price,
            gbm_delta=gbm_black_scholes_results.delta
        )
        return gbm_results

    def runSABRSimulation(self) -> SABRSimulationRunResults:
        """
        Runs the SABR simulation
        :return: Returns the stock price, call price and delta
        """
        sabr_results = self.computeSABRSimulation()
        sabr_implied_volatility = self.computeImpliedVolatility(
            stock_price=sabr_results.underlying_price,
            stochastic_volatility=sabr_results.stochastic_volatility
        )
        sabr_call_price, sabr_delta = self.computeBlackScholesCall(
            current_stock_price=sabr_results.underlying_price,
            current_volatility=sabr_results.stochastic_volatility
        )
        sabr_bartlett_delta = self.computeBartlettDelta(sabr_results.underlying_price, sabr_implied_volatility)

        results = SABRSimulationRunResults(
            sabr_stock_price=sabr_results.underlying_price,
            sabr_volatility=sabr_results.stochastic_volatility,
            sabr_implied_volatility=sabr_implied_volatility,
            sabr_call_price=sabr_call_price,
            sabr_delta=sabr_delta,
            sabr_bartlett_delta=sabr_bartlett_delta
        )
        return results

    def runHestonSimulation(self) -> HestonSimulationResults:
        """
        Runs the Heston simulation
        :return: Returns the stock price, and call option price
        """
        heston_model = HestonModel(
            s0=self._parameters.start_stock_price,
            strike=self._parameters.strike_price,
            r=self._parameters.risk_free_rate,
            q=self._parameters.dividend_rate,
            expiry=self._expiry_time,
            kappa=self._parameters.volatility_mean_reversion,
            theta=self._parameters.long_term_volatility,
            v0=self._parameters.start_volatility,
            vol_of_vol=self._parameters.volatility_of_volatility,
            rho=self._parameters.volatility_correlation,
            num_simulations=self._parameters.n_paths,
            num_time_steps=self._time_horizon
        )

        self._logger.info("Constructing the Monte Carlo simulation of Heston model..")
        result = heston_model.simulateHestonProcess()
        self._logger.info(f"result.stock_paths.shape: {result.stock_paths.shape}")
        self._logger.info(f"result.volatility_paths.shape: {result.volatility_paths.shape}")
        self._logger.info(f"result.option_price_paths.shape: {result.option_price_paths.shape}")
        return result

    def runHestonSimulationUsingJax(self) -> HestonSimulationResults:
        """
        Runs the Heston simulation (using Jax)
        :return: Returns the stock price, and call option price
        """
        heston_parameters = HestonParams(
            kappa=self._parameters.volatility_mean_reversion,
            theta=self._parameters.long_term_volatility,
            sigma=self._parameters.heston_vol_of_vol,
            rho=self._parameters.volatility_correlation,
            v0=self._parameters.heston_start_vol,
            mean=self._parameters.return_on_stock
        )

        heston_pricer_model = HestonOptionPricerWithJax(
            heston_params=heston_parameters,
            S_0=self._parameters.start_stock_price,
            V_0=self._parameters.heston_start_vol,
            K=self._parameters.strike_price,
            r=self._parameters.risk_free_rate,
            n_paths=self._parameters.n_paths,
            n_time_steps=self._time_horizon,
            time_to_expiry=self._expiry_time,
            seed=configs.RANDOM_SEED
        )

        result = heston_pricer_model.simulateHestonProcess()
        self._logger.info(f"result.stock_paths.shape: {result.stock_paths.shape}")
        self._logger.info(f"result.volatility_paths.shape: {result.volatility_paths.shape}")
        self._logger.info(f"result.option_price_paths.shape: {result.option_price_paths.shape}")
        self._logger.info(f"result.option_deltas.shape: {result.option_deltas.shape}")
        return result

    def computeHedgingStrategy(
            self,
            method: str,
            notional: int,
            delta: np.ndarray,
            bartlett_delta: Optional[np.ndarray] = None
    ) -> HedgingStrategyResults:
        """
        Implements delta hedging for GBM model, delta hedging and bartlett hedging for SABR model.
        :param method: Simulation method, "GBM" or "SABR"
        :param notional: Number of stocks the option is written on
        :param delta: Time series of the option BS delta until maturity (calculated from simulation)
        :param bartlett_delta: Time series of the option Bartlett - delta until maturity (calculated from simulation)
                               only in SABR case
        :return: Returns the following outputs:
                    trading:    time series of trading decisions under BS delta hedging
                    holding:    time series of holding level of the underlying, under BS delta hedging
                    trading_bartlett: time series of trading decisions under Bartlett delta hedging
                    holding_bartlett: time series of holding level of the underlying, under Bartlett delta hedging

        """
        results = self.computeHedgingStrategy_(
            method,
            notional,
            delta,
            bartlett_delta
        )
        return results

    @staticmethod
    def computeHedgingStrategy_(
            method: str,
            notional: int,
            delta: np.ndarray,
            bartlett_delta: Optional[np.ndarray] = None
    ) -> HedgingStrategyResults:
        """
        Implements delta hedging for GBM model, delta hedging and bartlett hedging for SABR model.
        :param method: Simulation method, "GBM" or "SABR"
        :param notional: Number of stocks the option is written on
        :param delta: Time series of the option BS delta until maturity (calculated from simulation)
        :param bartlett_delta: Time series of the option Bartlett - delta until maturity (calculated from simulation)
                               only in SABR case
        :return: Returns the following outputs:
                    trading:    time series of trading decisions under BS delta hedging
                    holding:    time series of holding level of the underlying, under BS delta hedging
                    trading_bartlett: time series of trading decisions under Bartlett delta hedging
                    holding_bartlett: time series of holding level of the underlying, under Bartlett delta hedging

        """
        trading_black_scholes = np.diff(delta, axis=1)
        trading_black_scholes = np.concatenate((delta[:, 0].reshape(-1, 1), trading_black_scholes), axis=1)
        trading_black_scholes *= notional
        holding_black_scholes = delta * notional

        trading_bartlett, holding_bartlett = None, None
        if method == "SABR":
            # sabr bartlett delta hedging
            trading_bartlett = np.diff(bartlett_delta, axis=1)
            trading_bartlett = np.concatenate((bartlett_delta[:, 0].reshape(-1, 1), trading_bartlett), axis=1)
            trading_bartlett *= notional
            holding_bartlett = bartlett_delta * notional

        results = HedgingStrategyResults(
            trading_black_scholes=trading_black_scholes,
            holding_black_scholes=holding_black_scholes,
            trading_bartlett=trading_bartlett,
            holding_bartlett=holding_bartlett
        )
        return results

    def computeNotionalAdjustedPnlProcess(
            self,
            underlying_price: np.ndarray,
            option_price: np.ndarray,
            holding: np.ndarray
    ) -> AdjustedPnlProcessResults:
        """
        Calculates the notional-adjusted Accounting PnL process for a portfolio of a short call option,
        the underlying, with proportional trading costs.
        :param underlying_price: Underlying price process
        :param option_price: Option price (adjusted for number of underlying)
        :param holding: Holding process of number of the underlying held at each period
        :return: Outputs are:
                 accounting_pnl : process of Accounting PnL
                 holding_lagged: lagged process of number of underlying held at each period
        """
        results = self.computeNotionalAdjustedPnlProcess_(
            underlying_price,
            option_price,
            holding,
            self._parameters.notional,
            self._parameters.strike_price,
            self._parameters.cost_per_traded_stock
        )

        return results

    @staticmethod
    def computeNotionalAdjustedPnlProcess_(
            underlying_price: np.ndarray,
            option_price: np.ndarray,
            holding: np.ndarray,
            notional: float,
            strike_price: float,
            cost_per_traded_stock: float
    ) -> AdjustedPnlProcessResults:
        """
        Calculates the notional-adjusted Accounting PnL process for a portfolio of a short call option,
        the underlying, with proportional trading costs.
        :param underlying_price: Underlying price process
        :param option_price: Option price (adjusted for number of underlying)
        :param holding: Holding process of number of the underlying held at each period
        :param notional: Notional of the portfolio
        :param strike_price: Strike price
        :param cost_per_traded_stock: Cost per traded stock
        :return: Outputs are:
                 accounting_pnl : process of Accounting PnL
                 holding_lagged: lagged process of number of underlying held at each period
        """
        # create lagged variables for APL
        option_price_lagged = np.roll(option_price, 1)
        option_price_lagged[:, 0] = np.nan  # the first element was p[-1], this has to be changed to NaN
        underlying_price_lagged = np.roll(underlying_price, 1)
        underlying_price_lagged[:, 0] = np.nan  # the first element was S[-1], this has to be changed to NaN
        holding_lagged = np.roll(holding, 1)
        holding_lagged[:, 0] = np.nan  # the first element was holding[-1], this has to be changed to NaN

        # accounting PnL
        term_1 = -(option_price - option_price_lagged)
        term_2 = holding_lagged * (underlying_price - underlying_price_lagged)
        term_3 = -cost_per_traded_stock * np.abs(underlying_price * (holding - holding_lagged))
        accounting_pnl = term_1 + term_2 + term_3

        term_4 = -(np.maximum((underlying_price[:, -1] - strike_price), 0)*notional
                             - option_price_lagged[:, -1])
        term_5 = holding_lagged[:, -1] * (underlying_price[:, -1] - underlying_price_lagged[:, -1])
        term_6 = -cost_per_traded_stock * np.abs(underlying_price[:, -1]
                                * (holding[:, -1] - holding_lagged[:, -1]))
        accounting_pnl[:, -1] = term_4 + term_5 + term_6

        accounting_pnl = np.nancumsum(accounting_pnl, axis=1)

        results = AdjustedPnlProcessResults(
            accounting_pnl=accounting_pnl,
            holding_lagged=holding_lagged
        )
        return results

    def evaluateAgainstClassicalHedging(
            self,
            accounting_pnl: np.ndarray,
            option_price: np.ndarray
    ) -> ClassicalHedgingResults:
        """
        Evaluates selected strategy against classical delta hedging method.
        :param accounting_pnl: Cumulative accounting PnL array of shape (paths, periods)
        :param option_price: Option prices of shape (paths, periods)
        :return: Returns the following outputs:
                evaluation_function: Evaluation function
                percentage_mean_ratio: Percentage mean ratio
                percentage_std_ratio: Percentage std ration
        """
        results = self.evaluateAgainstClassicalHedging_(
            accounting_pnl,
            option_price,
            self._parameters.stdev_coefficient,
            self._parameters.notional
        )

        return results

    @staticmethod
    def evaluateAgainstClassicalHedging_(
            accounting_pnl: np.ndarray,
            option_price: np.ndarray,
            stdev_coefficient: float,
            notional: float
    ) -> ClassicalHedgingResults:
        """
        Evaluates selected strategy against classical delta hedging method.
        :param accounting_pnl: Cumulative accounting PnL array of shape (paths, periods)
        :param option_price: Option prices of shape (paths, periods)
        :param stdev_coefficient: Stdev coefficient of determination
        :param notional: Notional value of portfolio
        :return: Returns the following outputs:
                evaluation_function: Evaluation function
                percentage_mean_ratio: Percentage mean ratio
                percentage_std_ratio: Percentage std ration
        """
        mean_cost = -np.nanmean(accounting_pnl, axis=1)  # negative rewards = costs
        std_cost = np.nanstd(accounting_pnl, axis=1)
        evaluation_function = mean_cost + stdev_coefficient * std_cost

        percentage_mean_ratio = np.mean(mean_cost / (notional * option_price[:, 0]))
        percentage_std_ratio = np.mean(std_cost / (notional * option_price[:, 0]))

        results = ClassicalHedgingResults(
            evaluation_function=evaluation_function,
            percentage_mean_ratio=percentage_mean_ratio,
            percentage_std_ratio=percentage_std_ratio
        )
        return results

    @property
    def parameters(self) -> Parameters:
        """
        Getter property of simulation parameters
        :return: Returns simulator parameters
        """
        return self._parameters





