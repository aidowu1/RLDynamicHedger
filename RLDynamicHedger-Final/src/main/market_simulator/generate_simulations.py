import numpy as np
import os

from src.main.market_simulator.simulator import MarketSimulator
from src.main.market_simulator.simulator_results import (
                                                         GBMSimulationResults,
                                                         SABRSimulationRunResults,
                                                         HestonSimulationResults
                                                        )
import src.main.configs_global as configs
from src.main.utility.settings_reader import SettingsReader
from src.main.market_simulator.parameters import Parameters
from src.main.utility.utils import Helpers
from src.main.utility.logging import Logger


class SimulationGenerator:
    """
    Generates the market simulation
    """
    def __init__(
            self,
            parameter_settings_filename: str = None,
            parameters: Parameters = None
    ):
        """
        Constructor
        :param parameter_settings_filename: Parameters settings filename
        :param parameters: Parameters object
        """
        self._logger = Logger().getLogger()
        if parameter_settings_filename is not None:
            parameter_settings_data = Helpers.getParameterSettings(parameter_settings_filename)
            self._parameters = Parameters(**parameter_settings_data)
        else:
            self._parameters = parameters
        n_trading_months = self._parameters.maturity_in_months
        trading_frequency = self._parameters.trading_frequency
        is_in_the_money = self._parameters.is_in_the_money
        self._data_path_part = f"{configs.DATA_ROOT_FOLDER}/{n_trading_months}month/{trading_frequency}d/{is_in_the_money}"
        os.makedirs(self._data_path_part, exist_ok=True)

        self._gbm_asset_simulation_data_path = f"{self._data_path_part}/{configs.GBM_ASSET_SIMULATION_FILE_NAME}"
        self._gbm_option_price_simulation_data_path = (f"{self._data_path_part}/"
                                                       f"{configs.GBM_OPTION_PRICE_SIMULATION_FILE_NAME}")
        self._gbm_option_delta_simulation_data_path = (f"{self._data_path_part}/"
                                                       f"{configs.GBM_OPTION_DELTA_SIMULATION_FILE_NAME}")

        self._sabr_asset_simulation_data_path = f"{self._data_path_part}/{configs.SABR_ASSET_SIMULATION_FILE_NAME}"
        self._sabr_option_price_simulation_data_path = (f"{self._data_path_part}/"
                                                        f"{configs.SABR_OPTION_PRICE_SIMULATION_FILE_NAME}")
        self._sabr_option_delta_simulation_data_path = (f"{self._data_path_part}/"
                                                        f"{configs.SABR_OPTION_DELTA_SIMULATION_FILE_NAME}")

        self._heston_asset_simulation_data_path = (f"{self._data_path_part}/"
                                                   f"{configs.HESTON_ASSET_SIMULATION_FILE_NAME}")
        self._heston_option_price_simulation_data_path = (f"{self._data_path_part}/"
                                                          f"{configs.HESTON_OPTION_PRICE_SIMULATION_FILE_NAME}")
        self._heston_option_delta_simulation_data_path = (f"{self._data_path_part}/"
                                                          f"{configs.HESTON_OPTION_DELTA_SIMULATION_FILE_NAME}")

        self._market_simulator = MarketSimulator(parameters=self._parameters)

    def generateGbmSimulation(self) -> bool:
        """
        Generates the GBM market simulation
        :return:
        """
        status = False
        try:

            gbm_results: GBMSimulationResults = self._market_simulator.runGBMSimulation()
            asset_price = gbm_results.gbm_stock_paths
            option_price = gbm_results.gbm_call_price

            np.savetxt(
                self._gbm_asset_simulation_data_path,
                asset_price,
                delimiter=configs.DELIMITER)

            np.savetxt(
                self._gbm_option_price_simulation_data_path,
                option_price,
                delimiter=configs.DELIMITER)

            np.savetxt(
                self._gbm_option_delta_simulation_data_path,
                gbm_results.gbm_delta,
                delimiter=configs.DELIMITER)
            status = True
        except Exception as e:
            error_msg = f"Error occurred during the GBM simulation generation\nError is:{str(e)}\n"
            self._logger.error(error_msg)
        return status


    def generateSabrSimulation(self):
        """
        Generates the SABR market simulation
        :return:
        """
        status = False
        try:
            sabr_results: SABRSimulationRunResults = self._market_simulator.runSABRSimulation()
            asset_price = sabr_results.sabr_stock_price
            option_price = sabr_results.sabr_call_price

            np.savetxt(
                self._sabr_asset_simulation_data_path,
                asset_price,
                delimiter=configs.DELIMITER)

            np.savetxt(
                self._sabr_option_price_simulation_data_path,
                option_price,
                delimiter=configs.DELIMITER)

            np.savetxt(
                self._sabr_option_delta_simulation_data_path,
                sabr_results.sabr_delta,
                delimiter=configs.DELIMITER)
            status = True
        except Exception as e:
            error_msg = f"Error occurred during the SABR simulation generation\nError is:{str(e)}\n"
            self._logger.error(error_msg)
        return status

    def generateHestonSimulation(self) -> bool:
        """
        Generates the Heston market simulation
        :return:
        """
        status = False
        try:

            heston_results: HestonSimulationResults = self._market_simulator.runHestonSimulationUsingJax()

            np.savetxt(
                self._heston_asset_simulation_data_path,
                heston_results.stock_paths,
                delimiter=configs.DELIMITER)

            np.savetxt(
                self._heston_option_price_simulation_data_path,
                heston_results.option_price_paths,
                delimiter=configs.DELIMITER)

            np.savetxt(
                self._heston_option_delta_simulation_data_path,
                heston_results.option_deltas,
                delimiter=configs.DELIMITER)
            status = True
        except Exception as e:
            error_msg = f"Error occurred during the Heston simulation generation\nError is:{str(e)}\n"
            self._logger.error(error_msg)
        return status




