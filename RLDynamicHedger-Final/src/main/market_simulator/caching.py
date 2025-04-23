import pandas as pd
import os
import numpy as np

from src.main import configs_global as configs
from src.main.market_simulator.parameters import Parameters
from src.main.market_simulator.generate_simulations import SimulationGenerator
from src.main.utility.enum_types import HedgingType
from src.main.utility.logging import Logger

class SimulationDataCache:
    """
    Simulation data cache for caching simulation data.
    """
    def __init__(
            self,
            parameters: Parameters
    ):
        """
        Constructor.
        :param Parameters parameters: Simulation parameters.
        """
        self._logger = Logger().getLogger()
        self._parameters = parameters

        self._asset_price_data_path = (f"{configs.DATA_ROOT_FOLDER}/{self._parameters.maturity_in_months}month/"
                                 f"{self._parameters.trading_frequency}d/{self._parameters.is_in_the_money}/"
                                 f"asset_price_{self._parameters.hedging_type.name}_simulation.csv")

        self._option_price_data_path = (f"{configs.DATA_ROOT_FOLDER}/{self._parameters.maturity_in_months}month/"
                                  f"{self._parameters.trading_frequency}d/{self._parameters.is_in_the_money}/"
                                  f"option_price_{self._parameters.hedging_type.name}_simulation.csv")

        self._option_delta_data_path = (f"{configs.DATA_ROOT_FOLDER}/{self._parameters.maturity_in_months}month/"
                                        f"{self._parameters.trading_frequency}d/{self._parameters.is_in_the_money}/"
                                        f"option_delta_{self._parameters.hedging_type.name}_simulation.csv")

    def generateSimulationResults(self):
        """
        Generates simulation result data.
        :return: Generation status
        """
        simulator = SimulationGenerator(parameters=self._parameters)
        match self._parameters.hedging_type:
            case HedgingType.gbm:
                status = simulator.generateGbmSimulation()
            case HedgingType.sabr:
                status = simulator.generateSabrSimulation()
            case HedgingType.heston:
                status = simulator.generateHestonSimulation()
            case _:
                raise ValueError("Invalid hedging type")
        return status

    @property
    def asset_price_data(self) -> np.ndarray:
        """
        Getter property for the asset price data.
        :return: Asset price data.
        """
        self._logger.info(f"Getting asset price data from {self._asset_price_data_path}")
        if os.path.exists(self._asset_price_data_path):
            try:
                asset_price_data = pd.read_csv(self._asset_price_data_path).values
            except Exception as e:
                error_msg = f"Error while reading asset price data: {e}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            self._logger.info(f"No asset price data found in the cache path: {self._asset_price_data_path}")
            self._logger.info("The data simulation data will need to be created from scratch, this will take a few minutes..")
            status = self.generateSimulationResults()
            if status:
                asset_price_data = pd.read_csv(self._asset_price_data_path).values
            else:
                error_msg = "Error while generating asset price data"
                raise ValueError(error_msg)
        return asset_price_data



    @property
    def option_price_data(self) -> np.ndarray:
        """
        Getter property for the option price data.
        :return: Option price data.
        """
        self._logger.info(f"Getting option price data from {self._option_price_data_path}")
        if os.path.exists(self._option_price_data_path):
            try:
                option_price_data = pd.read_csv(self._option_price_data_path).values
            except Exception as e:
                error_msg = f"Error while reading option price data: {e}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            self._logger.info(f"No option price data found in the cache path: {self._option_price_data_path}")
            self._logger.info(
                "The data simulation data will need to be created from scratch, this will take a few minutes..")
            status = self.generateSimulationResults()
            if status:
                option_price_data = pd.read_csv(self._option_price_data_path).values
            else:
                error_msg = "Error while generating option price data"
                raise ValueError(error_msg)
        return option_price_data

    @property
    def option_delta_data(self) -> np.ndarray:
        """
        Getter property for the option price data.
        :return: Option price data.
        """
        self._logger.info(f"Getting option delta data from {self._option_delta_data_path}")
        if os.path.exists(self._option_delta_data_path):
            try:
                option_delta_data = pd.read_csv(self._option_delta_data_path).values
            except Exception as e:
                error_msg = f"Error while reading option delta data: {e}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            self._logger.info(f"No option delta data found in the cache path: {self._option_delta_data_path}")
            self._logger.info(
                "The data simulation data will need to be created from scratch, this will take a few minutes..")
            status = self.generateSimulationResults()
            if status:
                option_delta_data = pd.read_csv(self._option_delta_data_path).values
            else:
                error_msg = "Error while generating option delta data"
                raise ValueError(error_msg)
        return option_delta_data

