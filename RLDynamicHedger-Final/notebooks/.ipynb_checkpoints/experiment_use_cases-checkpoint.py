import os, sys
import numpy as np
import time

SEED = 100
NEW_LINE = "\n"
np.random.seed(SEED)

ROOT_PATH = "../"
os.chdir(ROOT_PATH)
sys.path.insert(1, ROOT_PATH)
print(f"Current path is: {os.getcwd()}...{NEW_LINE}")

from src.main.utility.enum_types import PlotType, AggregationType, HedgingType, RLAgorithmType
from src.main.market_simulator.parameters import Parameters
from src.main.utility.utils import Helpers
from scripts.tune_hedger_rl_model import TuneHyperparametersForRLModels
import src.main.configs_global as configs

class RunScenarioBase:
    """
    Run Scenario base class
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        self.parameters = parameters
        self.is_test_env = is_test_env

class LowExpiryScenario(RunScenarioBase):
    """
    Low  expiry run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 63
        self.parameters.trading_frequency = 1.0
        self.parameters.option_expiry_time = 0.25
        self.parameters.maturity_in_months = int(12 * self.parameters.option_expiry_time)
        self.parameters.is_high_expiry_level = False     
        self.parameters.is_in_the_money = "ATM"
        self.parameters.is_test_env = is_test_env

class HighExpiryScenario(RunScenarioBase):
    """
    High expiry scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 252
        self.parameters.trading_frequency = 1.0
        self.parameters.option_expiry_time = 1
        self.parameters.maturity_in_months = int(12 * self.parameters.option_expiry_time)
        self.parameters.is_high_expiry_level = True
        self.parameters.is_in_the_money = "ATM"
        self.parameters.is_test_env = is_test_env

class LowTradingCostScenario(RunScenarioBase):
    """
    Low  transaction cost run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 63
        self.parameters.trading_frequency = 1.0
        self.parameters.option_expiry_time = 0.25
        self.parameters.tick_size = 0.01
        self.parameters.cost_per_traded_stock = 0.01
        self.parameters.is_in_the_money = "ATM"
        self.parameters.is_test_env = is_test_env

class HighTradingCostScenario(RunScenarioBase):
    """
    High trading cost run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 63
        self.parameters.trading_frequency = 1.0
        self.parameters.option_expiry_time = 0.25
        self.parameters.tick_size = 0.05
        self.parameters.cost_per_traded_stock = 0.05
        self.parameters.is_in_the_money = "ATM"
        self.parameters.is_test_env = is_test_env
        

class LowTradingFrequencyScenario(RunScenarioBase):
    """
    Low  trading frequency run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 252
        self.parameters.trading_frequency = 4
        self.parameters.option_expiry_time = 1.0
        self.parameters.maturity_in_months = int(12 * self.parameters.option_expiry_time)        
        self.parameters.frequency_level = "low"
        self.parameters.is_in_the_money = "ATM"
        self.parameters.is_test_env = is_test_env
        
class HighTradingFrequencyScenario(RunScenarioBase):
    """
    High frequency run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 252
        self.parameters.trading_frequency = 0.25
        self.parameters.option_expiry_time = 1.0
        self.parameters.maturity_in_months = int(12 * self.parameters.option_expiry_time)
        self.parameters.frequency_level = "high"
        self.parameters.is_in_the_money = "ATM"
        self.parameters.is_test_env = is_test_env

class HighMoneynessScenario(RunScenarioBase):
    """
    High moneyness run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 252
        self.parameters.trading_frequency = 1
        self.parameters.option_expiry_time = 1.0
        self.parameters.maturity_in_months = int(12 * self.parameters.option_expiry_time)
        self.parameters.start_stock_price = 110.0
        self.parameters.is_in_the_money = "ITM"
        self.parameters.is_test_env = is_test_env

class LowMoneynessScenario(RunScenarioBase):
    """
    Low moneyness run scenario
    """
    def __init__(
        self, 
        parameters: Parameters,
        is_test_env: bool = True    
    ):
        """
        Constructor
        :params parameters: Parameters
        :params is_test_env: Flag to indicate if the RL agent is being trained or tested
        """
        super().__init__(parameters, is_test_env)
        self.parameters.n_time_steps = 252
        self.parameters.trading_frequency = 1
        self.parameters.option_expiry_time = 1.0
        self.parameters.maturity_in_months = int(12 * self.parameters.option_expiry_time)
        self.parameters.start_stock_price = 80.0        
        self.parameters.is_in_the_money = "OTM"
        self.parameters.is_test_env = is_test_env

run_scenario_map = {
    "low_expiry": LowExpiryScenario,
    "high_expiry": HighExpiryScenario,
    "low_trading_cost": LowTradingCostScenario,
    "high_trading_cost": HighTradingCostScenario,
    "low_trading_freq": LowTradingFrequencyScenario,
    "high_trading_freq": HighTradingFrequencyScenario,
    "high_moneyness": HighMoneynessScenario,
    "low_moneyness": LowMoneynessScenario, 
}

def getRunScenarioParams(
    parameters: Parameters, 
    scenario: str = "low_trading_freq",
    is_test_env: bool = True
    
):
    """
    Gets the parameters used for a run scernario
    :param parameters: Default parameters
    :return: Run scenario parameters 
    """
    p = run_scenario_map.get(scenario)(parameters, is_test_env)
    return p.parameters
    