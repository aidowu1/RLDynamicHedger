from collections import namedtuple

BlackScholesCallResults = namedtuple("BlackScholesCallResults", ["price", "delta"])
SABRSimulationResults = namedtuple("SABRSimulationResults", ["underlying_price", "stochastic_volatility"])
GBMSimulationResults = namedtuple("GBMSimulationResults", ["gbm_stock_paths", "gbm_call_price", "gbm_delta"])
SABRSimulationRunResults = namedtuple("SABRSimulationRunResults",
                                   ["sabr_stock_price",
                                              "sabr_volatility",
                                              "sabr_implied_volatility",
                                              "sabr_call_price",
                                              "sabr_delta",
                                              "sabr_bartlett_delta"])
HedgingStrategyResults = namedtuple("HedgingStrategyResults",
                             ["trading_black_scholes",
                                        "holding_black_scholes",
                                        "trading_bartlett",
                                        "holding_bartlett"
                              ])
AdjustedPnlProcessResults = namedtuple("AdjustedPnlProcessResults",
                                       ["accounting_pnl", "holding_lagged"])
ClassicalHedgingResults = namedtuple("ClassicalHedgingResults",
                                     ["evaluation_function", "percentage_mean_ratio", "percentage_std_ratio"])

HestonSimulationResults = namedtuple(
    "HestonSimulationResults",
    ["stock_paths", "volatility_paths", "option_price_paths", "option_deltas"]
)