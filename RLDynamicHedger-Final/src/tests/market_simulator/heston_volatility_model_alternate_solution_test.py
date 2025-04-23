import unittest as ut
import inspect
import pathlib as p
import os

from src.main.market_simulator.heston_volatility_model_alternate_solution import HestonModel
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
        self.assertIsNotNone(heston_model, msg=error_msg)

    def test_HestonModel_Simulate_Heston_Process_Is_Valid(self):
        """
        Test the validity of the simulation of Heston path Monte Ca Carlo si.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
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
        self.assertIsNotNone(heston_model, msg=error_msg)
        result = heston_model.simulateHestonProcess(option_type=OptionType.call)
        self.assertIsNotNone(result, msg=error_msg)
        print(f"result.stock_paths.shape: {result.stock_paths.shape}")
        print(f"result.volatility_paths.shape: {result.volatility_paths.shape}")
        print(f"result.option_price_paths.shape: {result.option_price_paths.shape}")



if __name__ == '__main__':
    ut.main()
