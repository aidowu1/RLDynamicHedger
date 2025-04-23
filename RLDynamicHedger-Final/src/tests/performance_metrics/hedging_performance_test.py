import unittest as ut
import inspect
import os
import pandas as pd
from typing import List

from src.main.utility.logging import Logger
from src.main.performance_metrics.hedging_performance import HedgingPerformanceMetrics
from src.main.utility.enum_types import RLAgorithmType, HedgingType
import src.main.configs_rl as configs2
import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.market_simulator.parameters import Parameters
from src.main.performance_metrics.hedging_metrics import HullHedgingMetricsInputs

class HedgingPerformanceMetricsTest(ut.TestCase):
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
        self.td3_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.td3,
            self._hedging_type,
            episode=self.current_episode)
        self.ddpg_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.ddpg,
            self._hedging_type,
            episode=self.current_episode)
        self.ppo_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.ppo,
            self._hedging_type,
            episode=self.current_episode)
        self.sac_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.sac,
            self._hedging_type,
            episode=self.current_episode)



    def test_HedgingPerformanceMetrics_Constructor_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceMetrics" constructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceMetrics(
            td3_evaluation_results_df=self.td3_evaluation_results_df,
            ddpg_evaluation_results_df=self.ddpg_evaluation_results_df,
            ppo_evaluation_results_df=self.ppo_evaluation_results_df,
            sac_evaluation_results_df=self.sac_evaluation_results_df
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)
        print(f"Evaluation results for episode: {self.current_episode}")
        print(f"TD3 evaluation results rows and columns: {self.td3_evaluation_results_df.shape}")
        print(f"DDPG evaluation results rows and columns: {self.ddpg_evaluation_results_df.shape}")
        print(f"PPO evaluation results rows and columns: {self.ppo_evaluation_results_df.shape}")
        print(f"SAC evaluation results rows and columns: {self.sac_evaluation_results_df.shape}")

    def test_HedgingPerformanceMetrics_Compute_All_Metrics_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceMetrics" to compute all metrics
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceMetrics(
            td3_evaluation_results_df=self.td3_evaluation_results_df,
            ddpg_evaluation_results_df=self.ddpg_evaluation_results_df,
            ppo_evaluation_results_df=self.ppo_evaluation_results_df,
            sac_evaluation_results_df=self.sac_evaluation_results_df
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)
        hedging_metrics = hedging_performance.computeAllMetrics()
        self.assertIsNotNone(hedging_metrics, msg=error_msg)


    def test_HedgingPerformanceMetrics_Plot_Metrics_Is_Valid(self):
        """
        Test the validity of "HedgingPerformanceMetrics" to plot all metrics
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hedging_performance = HedgingPerformanceMetrics(
            td3_evaluation_results_df=self.td3_evaluation_results_df,
            ddpg_evaluation_results_df=self.ddpg_evaluation_results_df,
            ppo_evaluation_results_df=self.ppo_evaluation_results_df,
            sac_evaluation_results_df=self.sac_evaluation_results_df
        )
        self.assertIsNotNone(hedging_performance, msg=error_msg)
        hedging_metrics = hedging_performance.computeAllMetrics()
        self.assertIsNotNone(hedging_metrics, msg=error_msg)
        hedging_performance.plotMetrics(hedging_metrics)

    def _getEvaluationResultsPath(
            self,
            algorithm_type: RLAgorithmType,
            hedging_type: HedgingType,
            problem_title: str = "RL Delta Hedger"
    ):
        """
        Gets the RL evaluation results path
        :param algorithm_type: Algorithm type
        :param problem_title: Problem title
        :return:
        """
        joined_title = "_".join(problem_title.split())
        log_path = f"./logs/{joined_title}_{algorithm_type.name}_{hedging_type.name}"
        results_path = f"{log_path}/test_results/{self._hedging_type.name}.csv"
        return results_path

    def _getRLEvaluationResultsPerAlgorithmType(
            self,
            algorithm_type: RLAgorithmType,
            hedging_type: HedgingType,
            episode: int=0,
    ) -> pd.DataFrame:
        """
        Gets RL evaluation results for all algorithms
        :param algorithm_type: Algorithm type
        :param hedging_type: HedgingType
        :param episode: Episode
        :return:
        """
        results_path = self._getEvaluationResultsPath(algorithm_type, hedging_type)
        try:
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path,index_col=False)
                episode_filter = results_df.episode == episode
                episode_results_df = results_df[episode_filter]
                return episode_results_df
            else:
                raise Exception(f"{results_path} does not exist")
        except Exception as ex:
            self.logger.info(f"Exception: {ex}")

    def getAllRLEvaluationResultsForHullMetrics(
            self,
    ) -> List[HullHedgingMetricsInputs]:
        """
        Gets RL evaluation results for all algorithms/hedging types for Hull metrics
        :return:
        """
        all_metric_inputs = []
        algo_types = [RLAgorithmType.td3, RLAgorithmType.ddpg, RLAgorithmType.ppo, RLAgorithmType.sac]
        hedging_type = HedgingType.gbm
        for algorithm_type in algo_types:
            metric_inputs = Helpers.getRLEvaluationResultsForHullMetricsPerAlgorithmType(algorithm_type, hedging_type)
            all_metric_inputs.append(metric_inputs)
        return all_metric_inputs

if __name__ == '__main__':
    ut.main()
