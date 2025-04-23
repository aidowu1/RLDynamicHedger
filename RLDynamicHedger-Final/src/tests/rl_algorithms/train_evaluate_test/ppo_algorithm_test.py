import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy

from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.ppo_algorithm import PPOTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.ppo_hyper_parameter_tuning import PPOHyperParameterTuning
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, AggregationType, PlotType, HedgingType
from src.tests.rl_algorithms.train_evaluate_test.rl_algorithm_test_cycle import RLAlgorithmTestCycle


class PPOTrainAlgorithmTest(ut.TestCase):
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
        self.hedging_type = HedgingType.gbm
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        parameters = Parameters(**parameter_settings_data)
        self.env = DynamicHedgingEnv()
        self.rl_algorithm_type = RLAgorithmType.ppo

    def test_PPOTrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ppo_agent = PPOTrainAlgorithm(self.env)
        self.assertIsNotNone(ppo_agent, msg=error_msg)

    def test_PPOHyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = PPOHyperParameterTuning(
            self.env,
            self.rl_algorithm_type,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_PPOTrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 200
        ppo_agent = PPOTrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ppo_agent, msg=error_msg)
        ppo_agent.train()
        self.evaluateTrainedModel(ppo_agent.trained_model)


    def test_PPOTrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the DDPG RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 200
        ppo_agent = PPOTrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ppo_agent, msg=error_msg)
        ppo_agent.evaluate()
        print(f"The RL agent evaluation results are:\n {ppo_agent._evaluation_results_df.head()}")
        selected_columns = ["bs_delta", "rl_delta"]
        print(ppo_agent._evaluation_results_df[selected_columns].head(10))

    def test_DDPGTrainAlgorithm_RL_Agent_Test_Cycles_Is_Valid(self):
        """
        Test the validity of DDPG RL agent test cycle results.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        ppo_agent = PPOTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(ppo_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=ppo_agent,

        )
        results_df = rl_test_cycle.rlAgentTestRunAllCycles()
        self.assertIsNotNone(rl_test_cycle, msg=error_msg)
        print(f"Sample of RL test cycle results with a total of {results_df.shape[0]} rows:")
        print(results_df.head(10))

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

