import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy

from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.ddpg_algorithm import DDPGTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.ddpg_hyper_parameter_tuning import DDPGHyperParameterTuning
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, AggregationType, PlotType, HedgingType
from src.tests.rl_algorithms.train_evaluate_test.rl_algorithm_test_cycle import RLAlgorithmTestCycle


class DDPGTrainAlgorithmTest(ut.TestCase):
    """
    DDPG Network Test
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
        self.hedging_type = HedgingType.sabr
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        parameters = Parameters(**parameter_settings_data)
        self.env = DynamicHedgingEnv()
        self.rl_algorithm_type = RLAgorithmType.ddpg

    def test_DDPGTrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ddpg_agent = DDPGTrainAlgorithm(self.env, hedging_type=self.hedging_type)
        self.assertIsNotNone(ddpg_agent, msg=error_msg)

    def test_DDPGHyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = DDPGHyperParameterTuning(self.env, self.rl_algorithm_type, hedging_type=self.hedging_type)
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_DDPGTrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 1000
        ddpg_agent = DDPGTrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type = self.hedging_type
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)
        ddpg_agent.train()
        self.evaluateTrainedModel(ddpg_agent.trained_model)


    def test_DDPGTrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the DDPG RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 200
        ddpg_agent = DDPGTrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)
        #ddpg_agent.evaluate(model_path="./logs/RL_Delta_Hedger_ddpg/best_model.zip")
        ddpg_agent.evaluate()
        print(f"The RL agent evaluation results are:\n {ddpg_agent._evaluation_results_df.head()}")
        selected_columns = ["bs_delta", "rl_delta"]
        print(ddpg_agent._evaluation_results_df[selected_columns].head(10))

    def test_DDPGTrainAlgorithm_RL_Agent_Test_Cycles_Is_Valid(self):
        """
        Test the validity of DDPG RL agent test cycle results.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        ddpg_agent = DDPGTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=ddpg_agent,
        )
        results_df = rl_test_cycle.rlAgentTestRunAllCycles()
        self.assertIsNotNone(rl_test_cycle, msg=error_msg)
        print(f"Sample of RL test cycle results with a total of {results_df.shape[0]} rows:")
        print(results_df.head(10))

    def test_DDPGTrainAlgorithm_RL_Agent_Test_Cycles_Aggregate_Results_Plots_Is_Valid(self):
        """
        Test the validity of DDPG RL agent test cycle results plots.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        ddpg_agent = DDPGTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=ddpg_agent,
        )
        pnl_df, reward_df, trading_cost_df, delta_df = rl_test_cycle.aggregateResults(
            aggregation_type=AggregationType.mean
        )
        self.assertIsNotNone(pnl_df, msg=error_msg)
        self.assertIsNotNone(reward_df, msg=error_msg)
        self.assertIsNotNone(trading_cost_df, msg=error_msg)
        print(f"Sample of RL test cycle aggregate Pnl results with a total of {pnl_df.shape[0]} rows:")
        print(pnl_df.head(10))
        print(f"Sample of RL test cycle aggregate reward results with a total of {reward_df.shape[0]} rows:")
        print(reward_df.head(10))
        print(f"Sample of RL test cycle aggregate Trading Cost results with a total of {trading_cost_df.shape[0]} rows:")
        print(trading_cost_df.head(10))
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.rewards
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.pnl
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.trading_cost
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

