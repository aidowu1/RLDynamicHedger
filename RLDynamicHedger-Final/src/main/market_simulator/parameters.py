from dataclasses import dataclass

from src.main.utility.enum_types import HedgingType
import src.main.configs_global as configs
@dataclass
class Parameters:
    """
    Parameter settings for the market simulation
    """
    n_paths: int
    n_time_steps: int
    n_days_per_year: int
    trading_frequency: int
    option_expiry_time: int
    start_stock_price: float
    strike_price: float
    volatility: float
    start_volatility: float
    volatility_of_volatility: float
    risk_free_rate: float
    dividend_rate: float
    return_on_stock: float
    cost_per_traded_stock: float
    rho: float
    stdev_coefficient: float
    central_difference_spacing: float
    notional: int
    is_reset_path: bool
    is_test_env: bool
    hedging_type: HedgingType
    maturity_in_months: int
    n_business_days: int
    volatility_mean_reversion: float
    long_term_volatility: float
    volatility_correlation: float
    hedging_time_step: float
    trading_cost_parameter: float
    risk_averse_level: float
    is_include_option_price_feature: float
    epsilon: float
    tick_size: float
    is_in_the_money: str
    is_high_expiry_level: bool
    frequency_level: str
    evaluation_path_index: int
    heston_vol_of_vol: float
    heston_start_vol: float

    def __post_init__(self):
        # self.is_in_the_money = "T" if float(self.start_stock_price/self.strike_price) > 1.0 else "F"
        moneyness = float(self.start_stock_price/self.strike_price)
        if moneyness == 1.0:
            self.is_in_the_money = "ATM"
        elif moneyness > 1.0:
            self.is_in_the_money = "ITM"
        else:
            self.is_in_the_money = "OTM"

        if self.hedging_time_step == 1:
            self.hedging_time_step = float(self.hedging_time_step/self.n_days_per_year)

        if self.maturity_in_months == 0:
            self.maturity_in_months = int(self.option_expiry_time * configs.N_MONTHS_PER_YEAR)

        self.is_high_expiry_level = True if self.option_expiry_time >= 1 else False

        if self.trading_frequency < 1:
            self.frequency_level = "high"
        else:
            self.frequency_level = "low"



