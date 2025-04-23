from dataclasses import dataclass
import numpy as np


@dataclass
class HedgingMetrics:
    """
    Hedging metrics class
    """
    bs_mean_error: float
    bs_std_error: float
    td3_mean_error: float
    td3_std_error: float
    ddpg_mean_error: float
    ddpg_std_error: float
    ppo_mean_error: float
    ppo_std_error: float
    sac_mean_error: float
    sac_std_error: float

@dataclass
class HullHedgingMetrics:
    """
    Hull (approach) Hedging metrics class
    It is based on Hull et al (2020). "Deep hedging of derivatives using reinforcement learning".
    Journal of Financial Data Science, 3(1), 10–27
    """
    bs_mean_error: float
    bs_std_error: float
    bs_y_function: float
    td3_mean_error: float
    td3_std_error: float
    td3_y_function: float
    ddpg_mean_error: float
    ddpg_std_error: float
    ddpg_y_function: float
    ppo_mean_error: float
    ppo_std_error: float
    ppo_y_function: float
    sac_mean_error: float
    sac_std_error: float
    sac_y_function: float


@dataclass
class HullHedgingMetricsInputs:
    """
    Hull hedging metrics input class
    """
    bs_delta: np.ndarray
    rl_delta: np.ndarray
    asset_price: np.ndarray
    option_price: np.ndarray


