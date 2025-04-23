import unittest as ut
import inspect
import os

import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.market_simulator.caching import SimulationDataCache
from src.main.market_simulator.parameters import Parameters

class SimulationGeneratorTest(ut.TestCase):
    """
    Integration test suit for "simulationDataCache" component
    """
    def setUp(self):
        """
        Test set-up fixture
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        self._parameters = Parameters(**parameter_settings_data)

    def test_SimulationDataCache_Constructor_Is_Valid(self):
        """
        Test the validity of "SimulationDataCache" contructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulation_cache = SimulationDataCache(self._parameters)
        self.assertIsNotNone(simulation_cache, msg=error_msg)

    def test_SimulationDataCache_Get_Asset_Price_Data_Is_Valid(self):
        """
        Test the validity of "SimulationDataCache" to get the asset price data
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulation_cache = SimulationDataCache(self._parameters)
        self.assertIsNotNone(simulation_cache, msg=error_msg)
        asset_price_data = simulation_cache.asset_price_data
        self.assertIsNotNone(asset_price_data, msg=error_msg)
        print(f"Sample of asset price data is: {asset_price_data[:5]}")

    def test_SimulationDataCache_Get_Option_Price_Data_Is_Valid(self):
        """
        Test the validity of "SimulationDataCache" to get the option price data
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulation_cache = SimulationDataCache(self._parameters)
        self.assertIsNotNone(simulation_cache, msg=error_msg)
        option_price_data = simulation_cache.option_price_data
        self.assertIsNotNone(option_price_data, msg=error_msg)
        print(f"Sample of asset price data is: {option_price_data[:5]}")

    def test_SimulationDataCache_Get_Delta_Price_Data_Is_Valid(self):
        """
        Test the validity of "SimulationDataCache" to get the delta price data
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulation_cache = SimulationDataCache(self._parameters)
        self.assertIsNotNone(simulation_cache, msg=error_msg)
        option_delta_data = simulation_cache.option_delta_data
        self.assertIsNotNone(option_delta_data, msg=error_msg)
        print(f"Sample of asset price data is: {option_delta_data[:5]}")



