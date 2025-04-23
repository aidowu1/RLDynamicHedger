import unittest as ut
import inspect
import pathlib as p
import os

from src.main.market_simulator.heston_volatility_model_alternate_solution import HestonModel
from src.main.market_simulator.heston_volatility_model_jax_solution import HestonOptionPricerWithJax
from src.main.market_simulator.heston_parameters import HestonParams
from src.main.utility.utils import Helpers
import src.main.configs_global as configs
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import OptionType

class HestonModelTest(ut.TestCase):
    """
    Test suit for HestonModel class
    """
    def setUp(self):
        """
        Setup test fixture
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        self._parameters = Parameters(**parameter_settings_data)
        self._expiry_time = self._parameters.n_time_steps / self._parameters.n_days_per_year
        self._time_horizon = int(self._parameters.n_time_steps / self._parameters.trading_frequency)

    def test_HestonModel_Constructor_Is_Valid(self):
        """
        Test the validity of the "HestonModel" constructor.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        heston_parameters = HestonParams(
            kappa=self._parameters.volatility_mean_reversion,
            theta=self._parameters.long_term_volatility,
            sigma=self._parameters.heston_vol_of_vol,
            rho=self._parameters.volatility_correlation,
            v0=self._parameters.heston_start_vol,
            mean=self._parameters.return_on_stock
        )

        heston_pricer = HestonOptionPricerWithJax(
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

        self.assertIsNotNone(heston_pricer, msg=error_msg)

    def test_HestonModel_Simulate_Heston_Process_Is_Valid(self):
        """
        Test the validity of the simulation of Heston path Monte Carlo simulation.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        heston_parameters = HestonParams(
            kappa=self._parameters.volatility_mean_reversion,
            theta=self._parameters.long_term_volatility,
            sigma=self._parameters.heston_vol_of_vol,
            rho=self._parameters.volatility_correlation,
            v0=self._parameters.heston_start_vol,
            mean=self._parameters.return_on_stock
        )

        heston_pricer = HestonOptionPricerWithJax(
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

        self.assertIsNotNone(heston_pricer, msg=error_msg)
        result = heston_pricer.simulateHestonProcess()
        self.assertIsNotNone(result, msg=error_msg)
        print(f"result.stock_paths.shape: {result.stock_paths.shape}")
        print(f"result.volatility_paths.shape: {result.volatility_paths.shape}")
        print(f"result.option_price_paths.shape: {result.option_price_paths.shape}")
        print(f"result.option_deltas.shape: {result.option_deltas.shape}")



if __name__ == '__main__':
    ut.main()
