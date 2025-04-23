import unittest as ut
import inspect
import os
import pathlib as p
from tqdm import tqdm

import src.main.configs_global as configs
from src.main.environment.env import DynamicHedgingEnv
from src.main.utility.utils import Helpers


class DynamicHedgingEnvTest(ut.TestCase):
    """
    Test suit for the 'DynamicHedgingEnv' class.
    """
    def setUp(self):
        """
        Test set-up fixture
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

    def test_test_DynamicHedgingEnv_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the DynamicHedgingEnv object.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        env = DynamicHedgingEnv()
        self.assertIsNotNone(env, msg=error_msg)
        print(f"env.action_space: {env.action_space}")
        print(f"env.observation_space: {env.observation_space}")
        print(f"env.asset_price.shape: {env.asset_price_data.shape}")
        print(f"env.asset_price[:2]:\n{env.asset_price_data[:2]}")
        print(f"env.option_price.shape: {env.option_price_data.shape}")
        print(f"env.option_price[:2]:\n{env.option_price_data[:2]}")

    def test_DynamicHedgingEnv_Step_Is_Valid(self):
        """
        Test the validity of invoking the stepping in the "DynamicHedgingEnv" environment
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 10
        env = DynamicHedgingEnv()
        env.reset()
        self.assertIsNotNone(env, msg=error_msg)
        for i in range(n_episodes):
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            print(f"observation: {next_state}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"done: {truncated}")
            print(f"done: {info}")

    def test_DynamicHedgingEnv_Reset_Is_Valid(self):
        """
        Test the validity of invoking the resetting of the "DynamicHedgingEnv" environment
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 10
        env = DynamicHedgingEnv()
        self.assertIsNotNone(env, msg=error_msg)
        env.reset()

        for i in tqdm(range(n_episodes), desc="Episodes"):
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)

        states, info = env.reset()
        holdings, asset_price, option_expiry_time = states[0], states[1], states[2]
        self.assertIsNotNone(holdings, msg=error_msg)
        self.assertIsNotNone(asset_price, msg=error_msg)
        self.assertIsNotNone(option_expiry_time, msg=error_msg)
        print(f"holdings: {holdings}")
        print(f"asset_price: {asset_price}")
        print(f"option_expiry_time: {option_expiry_time}")





if __name__ == '__main__':
    ut.main()
