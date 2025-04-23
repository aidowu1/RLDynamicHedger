import pickle
from typing import List, Dict, Optional, Any
from itertools import islice
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.main import configs_global as configs
from src.main.utility.settings_reader import SettingsReader
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.performance_metrics.hedging_metrics import HullHedgingMetricsInputs
from src.main.market_simulator.parameters import Parameters

class Helpers:
    """
    Helper utilities
    """

    @staticmethod
    def getParameterSettings(
            parameter_settings_filename: str = configs.DEFAULT_SETTINGS_NAME
    ) -> Optional[Dict[str, Any]]:
        """
        Gets the parameter settings for the RL hedger
        :return: Returns the parameter settings for the RL hedger
        """
        parameter_settings_data = None
        parameter_settings = SettingsReader(parameter_settings_filename)
        file_exists = parameter_settings.file_exists
        if not file_exists:
            print(f"Settings file name {parameter_settings_filename} does not exist!!")
        else:
            parameter_settings_data = parameter_settings.read()
        return parameter_settings_data

    @staticmethod
    def createDirectoryIfItDoesNotExist(directory: str):
        """
        Create a directory if it does not exist
        :param directory: The directory to create
        :return:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def getPojectRootPath(project_name: str = configs.PROJECT_ROOT_PATH) -> str:
        """
        Gets the project root path
        :param project_name: The project name
        :return: Returns the project root path
        """
        path = os.path.dirname(os.path.realpath(__file__))
        while not str(path).endswith(str(project_name)):
            path = Path(path).parent
        return path

    @staticmethod
    def serialObject(
            data: Any,
            pickle_path: str
    ) -> None:
        """
        Serialize the data to a pickle file
        :param data: Data to serialize
        :param pickle_path: Pickle path
        :return: None
        """
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def deserializeObject(pickle_path: str) -> Any:
        """
        Deserialize the data from a pickle file
        :param pickle_path: Pickle path
        :return: Returns the deserialized data
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def filterDict(
            input_dict: Dict[str, Any],
            filter_list: List[Any]
    ) -> Dict[str, Any]:
        """
        Remove/filter a input dictionary based on a list
        :param input_dict: The input dictionary
        :param filter_list:
        :return: Filtered dictionary
        """
        output_dict = {
            key: input_dict[key]
            for key in input_dict if key not in filter_list
        }
        return output_dict

    @staticmethod
    def getEnumType(
            enum_type: Any,
            enum_type_name: str
    ) -> Any:
        """
        Gets the enum type based on the specified type name
        :param enum_type: Enum type
        :param enum_type_name: Enum name
        :return: Enum value
        """
        enum_dict = enum_type.__dict__
        enum_value = enum_dict.get(enum_type_name.lower())
        if enum_value is None:
            raise ValueError(f"Enum type {enum_type_name} does not exist!")
        return enum_value

    @staticmethod
    def getHedgingBenchmarkName(hedging_type: HedgingType) -> Dict[str, str]:
        """
        Gets the current benchmark hedging strategy name
        :param hedging_type: The hedging type
        :return: Map of current benchmark name
        """
        match hedging_type:
            case HedgingType.gbm:
                return {
                    "name": "GBM",
                    "delta_column_name": "bs_delta",
                    "option_price_column_name": "bs_option_price",
                }
            case HedgingType.sabr:
                return {
                    "name": "SABR",
                    "delta_column_name": "sabr_delta",
                    "option_price_column_name": "sabr_option_price",
                }
            case HedgingType.heston:
                return {
                    "name": "Heston",
                    "delta_column_name": "heston_delta",
                    "option_price_column_name": "heston_option_price",
                }
            case _:
                raise Exception("Invalid hedging_type!")

    @staticmethod
    def chunklistCollection(
            lst: List[Any],
            chunk_size: int
    ) -> List[List[Any]]:
        """
        Chunks a list collection
        :param lst: The list to chunk
        :param chunk_size:
        :return: Chunked list
        """
        it = iter(lst)  # Create an iterator
        return [
            list(islice(it, chunk_size))
            for _ in range((len(lst) + chunk_size - 1) // chunk_size)
        ]

    @staticmethod
    def chunkArray(
            lst: List[Any],
            chunk_size: int
    ) -> List[Any]:
        """
        Chunks a list collection
        :param lst: The list to chunk
        :param chunk_size:
        :return: Chunked array
        """
        array_list = np.array(lst, dtype=object)
        chunked_array = np.array_split(array_list, chunk_size)
        return chunked_array

    @staticmethod
    def getEvaluationResultsPath(
            algorithm_type: RLAgorithmType,
            hedging_type: HedgingType,
            problem_title: str = "RL Delta Hedger",
            extra_description: Optional[str] = None
    ):
        """
        Gets the RL evaluation results path
        :param algorithm_type: Algorithm type
        :param problem_title: Problem title
        :return:
        """
        tuned_model_root_path = f"model/trained-tuned-models/{algorithm_type.name}/{extra_description}/"
        results_folder_path = (f"{tuned_model_root_path}/{hedging_type.name}/test_results")
        results_file_path = f"{results_folder_path}/{hedging_type.name}.csv"
        return results_file_path

    @staticmethod
    def getRLEvaluationResultsForHullMetricsPerAlgorithmType(
            algorithm_type: RLAgorithmType,
            hedging_type: HedgingType,
            extra_description: Optional[str] = None,
    ) -> HullHedgingMetricsInputs:
        """
        Gets RL evaluation results for all algorithms/hedging types for Hull metrics
        :param algorithm_type: Algorithm type
        :param hedging_type: HedgingType,
        :param extra_description: Extra description of the test
        :return:
        """
        results_path = Helpers.getEvaluationResultsPath(
            algorithm_type=algorithm_type,
            hedging_type=hedging_type,
            extra_description=extra_description
        )
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        parameters = Parameters(**parameter_settings_data)
        try:
            if os.path.exists(results_path):
                print(f"Reading results from: {results_path}")
                results_df = pd.read_csv(results_path,index_col=False)
                hull_metric_columns = ["bs_delta", "rl_delta", "current_stock_price",  "current_option_price"]
                results_subset_df = results_df[hull_metric_columns]
                bs_delta = results_subset_df.bs_delta.values
                rl_delta = results_subset_df.rl_delta.values
                current_stock_price = results_subset_df.current_stock_price.values
                current_option_price = results_subset_df.current_option_price.values
                bs_delta_2d = bs_delta.reshape(parameters.n_paths, -1)
                rl_delta_2d = rl_delta.reshape(parameters.n_paths, -1)
                current_stock_price_2d = current_stock_price.reshape(parameters.n_paths, -1)
                current_option_price_2d = current_option_price.reshape(parameters.n_paths, -1)
                hull_metrics_input = HullHedgingMetricsInputs(
                    bs_delta_2d,
                    rl_delta_2d,
                    current_stock_price_2d,
                    current_option_price_2d
                )
                return hull_metrics_input
            else:
                raise Exception(f"{results_path} does not exist")
        except Exception as ex:
            print(f"Exception: {ex}")

    @staticmethod
    def text2Boolean(arg):
        """
        Parses a string to a boolean type
        :param arg: Input argument
        :return: Boolean value
        """
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            print("Invalid argument specified, had to set it to default value of 'True'")
            return True



