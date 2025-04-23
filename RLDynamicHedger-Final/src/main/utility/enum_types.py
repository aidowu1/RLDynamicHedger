from enum import Enum


class HedgingType(Enum):
    """
    Enumeration of hedging types
    """
    gbm = 1
    sabr = 2
    heston = 3

class OptionType(Enum):
    call = 1
    put = 2

class RLAgorithmType(Enum):
    """
    Enumeration of RL algorithm types
    """
    ddpg = 1
    td3 = 2
    sac = 3
    ppo = 4

class PlotType(Enum):
    """
    Enumeration of plot types
    """
    delta = 1
    option_price = 2
    pnl = 3
    rewards = 4
    trading_cost = 5

class AggregationType(Enum):
    """
    Enumeration of aggregation types
    """
    sum = 1
    mean = 2
