import unittest as ut
import inspect
import os
import pandas as pd
from typing import List

from src.main.utility.logging import Logger
from src.main.utility.enum_types import RLAgorithmType, HedgingType, AggregationType
import src.main.configs_rl as configs2
import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.market_simulator.parameters import Parameters
from generate_hedger_rl_results_single_model_special_case import GenerateRLModelSpecialCaseResults


class MyTestCase(ut.TestCase):
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.logger = Logger.getLogger()
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.algo_type = RLAgorithmType.td3
        self.is_recompute = True
        self.aggregation_type = AggregationType.mean
        self.model_path = "./model/trained-tuned-models/td3/high_expiry_{}_{}"
        self.extra_test_description = "high_expiry_option_tests"
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        self.parameters = Parameters(**parameter_settings_data)

    def test_GenerateRLModelSpecialCaseResults_Constructor_Is_Valid(self):
        """
        Test the validity of the 'GenerateRLModelSpecialCaseResults' constructor.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = GenerateRLModelSpecialCaseResults(
            algo_type=self.algo_type,
            is_recompute=self.is_recompute,
            aggregation_type=self.aggregation_type,
            model_path=self.model_path,
            extra_description=self.extra_test_description
        )
        self.assertIsNotNone(model, msg=error_msg)

    def test_GenerateRLModelSpecialCaseResults_Run_Is_Valid(self):
        """
        Test the validity of the 'GenerateRLModelSpecialCaseResults' execution.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = GenerateRLModelSpecialCaseResults(
            algo_type=self.algo_type,
            is_recompute=self.is_recompute,
            aggregation_type=self.aggregation_type,
            model_path=self.model_path,
            extra_description=self.extra_test_description
        )
        self.assertIsNotNone(model, msg=error_msg)
        model.run()


if __name__ == '__main__':
    ut.main()
