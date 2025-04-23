import gymnasium as gym
import numpy as np

class NormalizeEnv(gym.Wrapper):
    """
    A custom wrapper to normalize observations and scale actions.
    """
    def __init__(self, env):
        super(NormalizeEnv, self).__init__(env)
        self.obs_mean = 0
        self.obs_var = 1
        self.num_steps = 0
        self.action_scale = (env.action_space.high - env.action_space.low) / 2
        self.action_bias = (env.action_space.high + env.action_space.low) / 2

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        normalized_obs = self.normalize_observation(obs)
        # print(f"obs: {obs}\nnormalized_obs: {normalized_obs}")
        return normalized_obs

    def step(self, action):
        # Scale the action from [-1, 1] to the original action space range
        scaled_action = self.scale_action(action)
        truncated = False
        obs, reward, done, _, info = self.env.step(scaled_action)
        return self.normalize_observation(obs), reward, done, _, info

    def render(self, mode='human'):
        pass  # Optional visualization

    def close(self):
        super().close()

    def normalize_observation(self, obs):
        """
        Normalize observations using running mean and variance.
        """
        if self.num_steps == 0:
            self.obs_mean, self.obs_var = self.computeInitialObsStats()
        info = None
        if isinstance(obs, tuple):
            obs_array = obs[0]
            info = {}
        else:
            obs_array = obs
        self.num_steps += 1
        self.obs_mean = self.obs_mean + (obs_array - self.obs_mean) / self.num_steps
        self.obs_var = self.obs_var + ((obs_array - self.obs_mean) ** 2 - self.obs_var) / self.num_steps
        result_np_array =  (obs_array - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)
        if isinstance(obs, tuple):
            return result_np_array, info
        else:
            return result_np_array

    def computeInitialObsStats(self, sample_size: int=1000):
        """
        Compute initial observation statistics.
        :param sample_size:
        :return:
        """

        states = []
        for _ in range(sample_size):
            action = self.env.action_space.sample()
            new_state, reward, done, _, info = self.env.step(action)
            states.append(new_state)
        states_array = np.array(states)
        states_mean = np.mean(states_array, axis=0)
        states_var = np.var(states_array, axis=0)
        return states_mean, states_var


    def scale_action(self, action):
            """
            Scale action from [-1, 1] to the environment's action range.
            """
            return self.action_scale * action + self.action_bias
