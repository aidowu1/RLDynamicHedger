import unittest as ut
import inspect
import os
import pandas as pd
from typing import List

from src.main.utility.logging import Logger
from src.main.performance_metrics.hedging_performance_hull_metrics_single_agent import HedgingPerformanceHullMetricsSingleAgent
from src.main.utility.enum_types import RLAgorithmType, HedgingType
import src.main.configs_rl as configs2
import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.market_simulator.parameters import Parameters
from src.main.performance_metrics.hedging_metrics import HullHedgingMetricsInputs

class HedgingPerformanceMetricsSingleAgentTest(ut.TestCase):
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.logger = Logger.getLogger()
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.env_name = "RL Delta Hedger"
        self._hedging_type = HedgingType.gbm
        self.current_episode = 0
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        self.parameters = Parameters(**parameter_settings_data)

    def test_HedgingPerformanceHullMetrics_Constructor_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceHullMetrics" constructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceHullMetricsSingleAgent(
            hedge_type=HedgingType.gbm,
            algo_type=RLAgorithmType.td3
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)

    def test_HedgingPerformanceHullMetrics_Tabulate_Results_GBM_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceHullMetrics" constructor (GBM case)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceHullMetricsSingleAgent(
            hedge_type=HedgingType.gbm,
            algo_type=RLAgorithmType.td3
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)
        results_df = hedging_performance.tabulateAllHullMetrics()
        self.assertIsNotNone(results_df, msg=error_msg)
        print("Hull metrics per strategy:")
        print(results_df)


    def test_HedgingPerformanceHullMetrics_Tabulate_Results_SABR_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceHullMetrics" constructor (SABR case)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceHullMetricsSingleAgent(
            hedge_type=HedgingType.sabr,
            algo_type=RLAgorithmType.td3
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)
        results_df = hedging_performance.tabulateAllHullMetrics()
        self.assertIsNotNone(results_df, msg=error_msg)
        print("Hull metrics per strategy:")
        print(results_df.head())

    def test_HedgingPerformanceHullMetrics_Tabulate_Results_Heston_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceHullMetrics" constructor (Heston case)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceHullMetricsSingleAgent(
            hedge_type=HedgingType.heston,
            algo_type=RLAgorithmType.td3
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)
        results_df = hedging_performance.tabulateAllHullMetrics()
        self.assertIsNotNone(results_df, msg=error_msg)
        print("Hull metrics per strategy:")
        print(results_df)

