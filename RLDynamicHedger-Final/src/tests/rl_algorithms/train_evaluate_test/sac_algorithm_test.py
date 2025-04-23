import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy


from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.sac_algorithm import SACTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.sac_hyper_parameter_tuning import SACHyperParameterTuning
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, AggregationType, PlotType, HedgingType
from src.tests.rl_algorithms.train_evaluate_test.rl_algorithm_test_cycle import RLAlgorithmTestCycle


class SACTrainAlgorithmTest(ut.TestCase):
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
        self.rl_algorithm_type = RLAgorithmType.sac

    def test_SACTrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the SAC RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        sac_agent = SACTrainAlgorithm(
            self.env,
            hedging_type=self.hedging_type
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)

    def test_SACHyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the SAC RL algorithm hyperparameter tuning.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = SACHyperParameterTuning(
            self.env,
            self.rl_algorithm_type,
            hedging_type=HedgingType.heston
        )
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_SACTrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of SAC RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 3000
        sac_agent = SACTrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.gbm
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        sac_agent.train()
        self.evaluateTrainedModel(sac_agent.trained_model)


    def test_SACTrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the SAC RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        sac_agent = SACTrainAlgorithm(
            self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.heston
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        model_path = f"{sac_agent.createSaveModelPath()}.zip"
        sac_agent.evaluate(model_path=model_path)
        #print(f"The RL agent evaluation results are:\n {sac_agent.evaluation_results_df.head()}")
        selected_columns = ["bs_delta", "rl_delta"]
        print(sac_agent._evaluation_results_df[selected_columns].head(10))

    def test_SACTrainAlgorithm_RL_Agent_Test_Cycles_Is_Valid(self):
        """
        Test the validity of SAC RL agent test cycle results.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        sac_agent = SACTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.gbm
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=sac_agent,

        )
        results_df = rl_test_cycle.rlAgentTestRunAllCycles()
        self.assertIsNotNone(rl_test_cycle, msg=error_msg)
        print(f"Sample of RL test cycle results with a total of {results_df.shape[0]} rows:")
        print(results_df.head(10))

    def test_SACTrainAlgorithm_RL_Agent_Test_Cycles_Aggregate_Results_Calculation_Is_Valid(self):
        """
        Test the validity of SAC RL agent test cycle results calculation.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        sac_agent = SACTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.gbm
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=sac_agent,
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

    def test_SACTrainAlgorithm_RL_Agent_Test_Cycles_Single_Episode_Results_Calculation_Is_Valid(self):
        """
        Test the validity of SAC RL agent test cycle results calculation for a single episode.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        sac_agent = SACTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.gbm
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=sac_agent,
        )
        pnl_df, reward_df, trading_cost_df, delta_df = rl_test_cycle.getSingleEpisodeResults()
        self.assertIsNotNone(pnl_df, msg=error_msg)
        self.assertIsNotNone(reward_df, msg=error_msg)
        self.assertIsNotNone(trading_cost_df, msg=error_msg)
        print(f"Sample of RL test cycle aggregate Pnl results with a total of {pnl_df.shape[0]} rows:")
        print(pnl_df.head(10))
        print(f"Sample of RL test cycle aggregate reward results with a total of {reward_df.shape[0]} rows:")
        print(reward_df.head(10))
        print(f"Sample of RL test cycle aggregate Trading Cost results with a total of {trading_cost_df.shape[0]} rows:")
        print(trading_cost_df.head(10))

    def test_SACTrainAlgorithm_RL_Agent_Test_Cycles_Aggregate_Results_Plots_Is_Valid(self):
        """
        Test the validity of SAC RL agent test cycle results plots.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        sac_agent = SACTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.gbm
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=sac_agent,
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

    def test_SACTrainAlgorithm_RL_Agent_Test_Cycles_Single_Episode_Results_Plots_Is_Valid(self):
        """
        Test the validity of SAC RL agent test cycle results plots for a single episode.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 400
        sac_agent = SACTrainAlgorithm(
            env=self.env,
            total_timesteps=configs2.N_STEPS * n_episodes,
            hedging_type=HedgingType.gbm
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self.env,
            agent=sac_agent,
        )
        pnl_df, reward_df, trading_cost_df, delta_df = rl_test_cycle.getSingleEpisodeResults()
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
            plot_type=PlotType.rewards,
            is_single_episode=True
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.pnl,
            is_single_episode=True
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.trading_cost,
            is_single_episode=True
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.delta,
            is_single_episode=True
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

