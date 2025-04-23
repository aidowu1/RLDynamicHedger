import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy

from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.td3_algorithm import TD3TrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning import TD3HyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, AggregationType, PlotType, HedgingType
from src.tests.rl_algorithms.train_evaluate_test.rl_algorithm_test_cycle import RLAlgorithmTestCycle


class TD3TrainAlgorithmTest(ut.TestCase):
    """
    TD3 Network Test
    """
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.env_name = "RL Delta Hedger"
        self.hedging_type = HedgingType.gbm
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        self._parameters = Parameters(**parameter_settings_data)
        self.env = DynamicHedgingEnv()
        self.rl_algorithm_type = RLAgorithmType.td3
        self._expiry_level = "high" if self._parameters.is_high_expiry_level else "low"
        self._trained_model_path = (f"./model/trained-tuned-models/td3/"
                                   f"{self._expiry_level}_expiry_{self._parameters.hedging_type.name}_"
                                   f"{self.rl_algorithm_type.name}/best_model.zip")

    def test_TD3TrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ddpg_agent = TD3TrainAlgorithm(
            self.env,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)

    def test_TD3HyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = TD3HyperParameterTuning(
            self.env,
            self.rl_algorithm_type,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_TD3TrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 500
        td3_agent = TD3TrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_TUNING_TRAIN_STEPS,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(td3_agent, msg=error_msg)
        td3_agent.train()
        self.evaluateTrainedModel(td3_agent.trained_model)

    def test_TD3TrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the TD3 RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        td3_agent = TD3TrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_TUNING_TRAIN_STEPS,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(td3_agent, msg=error_msg)
        results_df = td3_agent.evaluate(model_path=self._trained_model_path)
        print(f"The RL agent evaluation results are:\n {results_df.head()}")
        selected_columns = ["bs_delta", "rl_delta", "bs_option_price", "rl_option_price"]
        print(results_df[selected_columns].head(10))
        model = td3_agent.trained_model
        self.evaluateTrainedModel(model)

    def test_TD3TrainAlgorithm_RL_Agent_Test_Cycles_Is_Valid(self):
        """
        Test the validity of TD3 RL agent test cycle results.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        td3_agent = TD3TrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(td3_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=td3_agent,

        )
        results_df = rl_test_cycle.rlAgentTestRunAllCycles()
        self.assertIsNotNone(rl_test_cycle, msg=error_msg)
        print(f"Sample of RL test cycle results with a total of {results_df.shape[0]} rows:")
        print(results_df.head(10))

    def test_TD3TrainAlgorithm_RL_Agent_Test_Cycles_Aggregate_Results_Plots_Is_Valid(self):
        """
        Test the validity of TD3 RL agent test cycle results plots.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        td3_agent = TD3TrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(td3_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=td3_agent,
        )
        pnl_df, reward_df, trading_cost_df, delta_df  = rl_test_cycle.aggregateResults(
            aggregation_type=AggregationType.sum
        )
        self.assertIsNotNone(pnl_df, msg=error_msg)
        self.assertIsNotNone(reward_df, msg=error_msg)
        self.assertIsNotNone(trading_cost_df, msg=error_msg)
        self.assertIsNotNone(delta_df, msg=error_msg)
        print(f"Sample of RL test cycle aggregate Pnl results with a total of {pnl_df.shape[0]} rows:")
        print(pnl_df.head(10))
        print(f"Sample of RL test cycle aggregate reward results with a total of {reward_df.shape[0]} rows:")
        print(reward_df.head(10))
        print(f"Sample of RL test cycle aggregate Trading Cost results with a total of {trading_cost_df.shape[0]} rows:")
        print(trading_cost_df.head(10))
        print(
            f"Sample of RL test cycle aggregate Delta results with a total of {delta_df.shape[0]} rows:")
        print(delta_df.head(10))
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.rewards
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.pnl
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.trading_cost
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.delta
        )

        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.rewards
        )
        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.pnl
        )
        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.trading_cost
        )
        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.delta
        )

    def evaluateTrainedModel(self, model):
        """
        Evaluates a trained model
        :param model: Model
        :return: None
        """
        mean_reward, std_reward = evaluate_policy(
            model,
            self.env,
            n_eval_episodes=10,
            deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
