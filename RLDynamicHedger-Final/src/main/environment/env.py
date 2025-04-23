import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from typing import Dict, Any, List

from src.main import configs_global as configs
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import HedgingType
from src.main.utility.utils import Helpers
from src.main.market_simulator.caching import SimulationDataCache
from src.main.utility.logging import Logger

class DynamicHedgingEnv(gym.Env):
    """
    RL Dynamic Hedging Environment
    """
    def __init__(
        self,
        parameter_settings_filename: str = configs.DEFAULT_SETTINGS_NAME,
        hedging_type: HedgingType = None,
        parameters: Parameters = None,
    ):
        """
        Constructor
        """
        self._logger = Logger.getLogger()
        self.name = "RLDynamicHedger"

        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(parameter_settings_filename)
            self._parameters = Parameters(**parameter_settings_data)
        if hedging_type:
            self._parameters.hedging_type = hedging_type
        self._logger.info(f"parameters:\n{self._parameters}\n")
        self._applyAssertions()


        self.simulation_cache = SimulationDataCache(self._parameters)
        self._asset_price_data = self.simulation_cache.asset_price_data
        self._option_price_data = self.simulation_cache.option_price_data
        self._option_delta_data = self.simulation_cache.option_delta_data

        self._n_paths = self._option_price_data.shape[0]
        self._n_steps = self._option_price_data.shape[1]

        # Initialize replication portfolio, V(t) components:
        #  - B(t): Money market account
        #  - X(t): Number of replicating portfolio shares
        #  So V(t) = B(t) + X(t)
        #  Thus, the option (call) payoff, c(t) is replicated as follows: V(t) = c(t)
        self._hedging_portfolio_value = 0.0
        self._current_hedging_delta = 0.0
        self._money_market_account_value = 0.0

        # Compute the Time To Maturity vector (TTM aka tau)
        self._tau = self._parameters.option_expiry_time - np.linspace(
            0.0,
            self._parameters.option_expiry_time,
            self._n_steps
        )

        # user-defined options (path)
        self._reset_path = self._parameters.is_reset_path
        self._path_choice = int(random.uniform(0, self._n_paths))
        self._path_index = 0

        # initialize price memory for normalization
        self._price_memory = []
        self._price_stat = []
        self._window_length = configs.PRICE_NORMALIZATION_WINDOW

        # Current change in holdings (i.e. change in delta)
        self._current_delta_change = 0

        # Current Pnl and reward values
        self._current_pnl = 0.0
        self._current_reward = 0.0

        # transaction cost (for rewards)
        self._kappa = self._parameters.cost_per_traded_stock

        # initializing underlying asset and option details
        self._current_step = 0
        self._notional = self._parameters.notional
        self._strike = self._parameters.strike_price

        # Actions of the format hold amount [-1,0]
        self._action_space = spaces.Box(low=-1.0, high=0.0, shape=(1,))

        if not self._parameters.is_include_option_price_feature:
            # Setup 3-D State, agent is given previous action + current asset price and time to maturity (H_i-1, S_i, tau_i)
            self._observation_space = spaces.Box(
                low=np.array([-1, 0, 0]),
                high=np.array([1, np.inf, 1]),
                shape=(3,),
                dtype=np.float32,
            )
        else:
            # Setup 5-D state, agent is given previous action,
            # current asset price, time to maturity, option price
            # and bs_delta (H_i-1, S_i, tau_i, C_i, bs_delta_i)
            self._observation_space = spaces.Box(
                low=np.array([-1, 0, 0, 0, 0]),
                high=np.array([1, np.inf, 1, 1, 1]),
                shape=(5,),
                dtype=np.float32,
            )

        self._is_run_test = self._parameters.is_test_env
        self._rewards_for_env_episodes = None

    def step(self, action: float):
        """
        Method call used to step through the environment
        :param action:
        :return:
        """
        if isinstance(action, np.ndarray):
            current_action = action[0]
            # current_action = -action[0]
            # current_action = np.clip(0.5 * (action[0] + 1), 0, 1)
        else:
            current_action = action
            # current_action = -action
            # current_action = np.clip(0.5 * (action+ 1), 0, 1)

        # Execute one time step within the environment
        self._current_step += 1

        # next call price, call price now, next asset price, asset price now.
        self._option_price_now = self._option_price_data[self._path_index, self._current_step]
        self._option_price_previous = self._option_price_data[self._path_index, self._current_step - 1]

        self._stock_price_now = self._asset_price_data[self._path_index, self._current_step]
        self._stock_price_previous = self._asset_price_data[self._path_index, self._current_step - 1]

        self._option_delta_now = self._option_delta_data[self._path_index, self._current_step]
        self._option_delta_previous = self._option_delta_data[self._path_index, self._current_step - 1]

        # Initial values for feature normalization
        self._initial_asset_price = self._asset_price_data[self._path_index, self._current_step]
        self._initial_option_price = self._option_price_data[self._path_index, self._current_step]

        # Compute the current transaction cost, Pnl and reward values
        self._current_reward = self._computeRewardFunction()

        # Update the hedging portfolio value
        self._hedging_portfolio_value = self._calculateHedgingReplicationPortfolioValue(current_action)

        # done: whether the episode is ended or not
        if self._current_step + 1 >= self._n_steps:
            done = True
        else:
            done = False

        truncated = False

        next_state = self._getStates()
        info = self._getInfos()

        reward = float(self._current_reward)
        self._rewards_for_env_episodes.append(reward)

        self._updateDeltaValue(current_action)
        return next_state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Method call used to reset the environment
        :return:
        """
        if not self._rewards_for_env_episodes:
            self._rewards_for_env_episodes = []

        info = {}
        if self._is_run_test:
            # self._path_index = -1 if self._path_index == 0 else self._path_index
            self._path_index += 1
            if self._path_index + 1 >= self._asset_price_data.shape[0]:
                self._path_index = 0
        elif self._reset_path:  # if user chose True, sets to his choice, else, random
                self._path_index = self._path_choice
        else:
            if self._parameters.evaluation_path_index > -1:
                self._path_index = self._parameters.evaluation_path_index

        # when resetting the env, set current_step and previous holdings equal to 0.
        self._current_step = 0

        # Current change in holdings
        self._current_delta_change = 0

        # Initialize replicating portfolio (synthetic option)
        self._hedging_portfolio_value = self._option_price_data[self._path_index, self._current_step]
        self._current_hedging_delta = -self._option_delta_data[self._path_index, self._current_step]
        self._money_market_account_value = -(
                self._hedging_portfolio_value + self._current_hedging_delta * self._asset_price_data[self._path_index, 0]
        )

        states = self._getStates()
        return states, info

    def render(self, mode='human'):
        pass  # Optional visualization

    def close(self):
        self._rewards_for_env_episodes = None
        super().close()

    def _updateDeltaValue(self, new_delta: float) -> None:
        """
        Updates the change in delta
        :param new_delta: New delta value (action)
        :return: Change in delta value
        """
        self._current_delta_change = new_delta - self._current_hedging_delta
        self._previous_holdings = self._current_hedging_delta
        self._current_hedging_delta = new_delta

    def _computeTransactionCost(
            self,
            change_in_holdings: float
    ) -> float:
        """
        Computes the transaction cost
        :param change_in_holdings: Change in holdings
        :return: Current transaction cost
        """
        part_1 = np.abs(change_in_holdings)
        part_2 = 0.01 * np.power(change_in_holdings, 2.0)
        current_transaction_cost = (self._parameters.epsilon * self._parameters.tick_size *
                                    (part_1 + part_2))

        return current_transaction_cost

    def _computePnLFunction(
            self
    ) -> float:
        """
        Computes the current PnL cost
        :param action: Action
        :return: Current PnL cost
        """
        # pnl_part_1 = (self._option_price_now - self._option_price_previous)
        # pnl_part_2 = self._current_hedging_delta * (self._stock_price_now - self._stock_price_previous)
        # pnl_part_3 = self._parameters.tick_size * np.abs(self._current_delta_change) * self._stock_price_now
        #
        # pnl = pnl_part_1 + pnl_part_2 - pnl_part_3
        dv = (self._option_price_now - self._option_price_previous)
        ds = (self._stock_price_now - self._stock_price_previous)
        pnl = dv + self._current_hedging_delta * ds - self._computeTransactionCost(self._current_delta_change)
        return pnl

    def _computeRewardFunction(
            self
    ) -> float:
        """
        Computes the reward function. The periodical reward,approximated facilitates a mean-variance
        optimization through RL training
        :return: Reward at the current time step
        """
        self._current_pnl = self._computePnLFunction()
        reward = self._current_pnl - (self._parameters.risk_averse_level / 2) * (self._current_pnl ** 2)
        return reward

    def _calculateHedgingReplicationPortfolioValue(
            self,
            action: float):
        """
        Calculates the hedging portfolio value
        :param action: Current RL agent action (newly computed delta)
        :return:
        """
        new_hedging_port_value = (
                self._money_market_account_value
                + self._current_hedging_delta * self._asset_price_data[self._path_index, self._current_step]
        )
        transaction_cost = self._computeTransactionCost(self._current_delta_change)
        self._money_market_account_value = (
                new_hedging_port_value
                - action * self._asset_price_data[self._path_index, self._current_step]
                - transaction_cost
        )
        return -new_hedging_port_value

    def _getStates(self) -> np.ndarray:
        """
        Get the RL states
        :return: RL states
        """
        bs_delta = self._option_delta_data[self._path_index, self._current_step]

        # Normalizing features
        asset_price_normalized = np.log(self._asset_price_data[self._path_index, self._current_step] / self._strike)
        option_price_normalized = self._option_price_data[self._path_index, self._current_step] / \
                                  self._option_price_data[self._path_index, 0]

        if not self._parameters.is_include_option_price_feature:
            # Return 3-D state: {H_i-1, S_i, tau_i}
            state = np.array([
                self._current_hedging_delta,
                asset_price_normalized,
                self._parameters.option_expiry_time,
            ], dtype=np.float32)
        else:
            # Return 5-D state: {H_i-1, S_i, tau_i, C_i, bs_delta_i}
            state = np.array([
                self._current_hedging_delta,
                asset_price_normalized,
                self._parameters.option_expiry_time,
                option_price_normalized,
                bs_delta,
            ], dtype=np.float32)
        return state

    def _getInfos(self) -> Dict[str, Any]:
        """
        Gets the RL infos
        :return: RL infos
        """
        info = {
            "current_transaction_cost": self._computeTransactionCost(self._current_delta_change),
            "current_pnl": self._current_pnl,
            "hedge_portfolio_value": self._hedging_portfolio_value,
            "money_market_account": self._money_market_account_value,
            "bs_delta": self._option_delta_data[self._path_index, self._current_step],
            "rl_delta": self._current_hedging_delta,
            "current_stock_price": self._asset_price_data[self._path_index, self._current_step],
            "current_option_price": self._option_price_data[self._path_index, self._current_step],
            "simulation_path_index": int(self._path_index)
        }
        return info


    def _applyAssertions(self):
        """
        Apply required assertions for the environment parameters
        :return:
        """
        assert self._parameters.hedging_type in [
            HedgingType.gbm,
            HedgingType.sabr,
            HedgingType.heston
        ], 'Wrong hedging_type input! Try one of these: {"GBM", "SABR", "Heston"}'
        assert self._parameters.maturity_in_months in configs.OPTION_MATURITY_MONTHS, \
            f"Wrong selection! Try any of the following correct option maturities: {configs.OPTION_MATURITY_MONTHS}"
        assert self._parameters.trading_frequency in configs.OPTION_TRADING_FREQUENCY, \
            f"Wrong selection! Try any of the following correct trading frequencies: {configs.OPTION_TRADING_FREQUENCY}"

    @property
    def option_price_data(self) -> np.ndarray:
        """
        Getter property for option price data
        :return: Returns option price
        """
        return self._option_price_data

    @property
    def asset_price_data(self) -> np.ndarray:
        """
        Getter property for asset (stock) price data
        :return: Returns asset (stock) price
        """
        return self._asset_price_data

    @property
    def action_space(self) -> gym.spaces.box.Box:
        """
        Getter for the action space
        :return: Returns the action space
        """
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.box.Box:
        """
        Getter for the observation space
        :return: Returns the observation space
        """
        return self._observation_space

    @property
    def is_run_test(self):
        """
        Getter for the is_run_test
        :return: Run test flag value
        """
        return self._is_run_test

    @is_run_test.setter
    def is_run_test(self, value):
        """
        Setter for is_run_test
        :param value:
        :return: None
        """
        self._is_run_test = value

    @property
    def reward_for_env_episodes(self) -> List[float]:
        """
        Getter property for all the rewards of environment episodes
        :return: Reward for all the rewards of environment episodes
        """
        return self._rewards_for_env_episodes

    @property
    def n_simulation_time_steps(self) -> int:
        """
        Getter property for number of simulation time steps
        :return: Number of simulation time steps
        """
        return self._n_steps - 1

    @property
    def parameters(self) -> Parameters:
        """
        Getter for the RL environment parameters
        :return: RL environment parameters
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Parameters) -> None:
        """
        Setter for the parameters attribute
        :param parameters: Input parameters
        :return: None
        """
        self._parameters = parameters

