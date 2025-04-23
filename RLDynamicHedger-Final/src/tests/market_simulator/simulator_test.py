import unittest as ut
import inspect
import numpy as np
import os
import pathlib as p


from src.main.market_simulator.simulator import MarketSimulator
from src.main.market_simulator.caching import SimulationDataCache
import src.main.configs_global as configs
from src.main.utility.utils import Helpers


class MarketSimulatorTest(ut.TestCase):
    """
    Test suit for the 'MarketSimulator' class.
    """
    def setUp(self):
        """
        Test set-up fixture
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.line_divider = "=====" * 10


    def test_MarketSimulator_Parameter_Less_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the MarketSimulator object (using parameterless constructor).
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator()
        self.assertIsNotNone(market_simulator, msg=error_msg)

    def test_MarketSimulator_Constructor_With_Config_Parameter_Is_Valid(self):
        """
        Test the validity of constructing the MarketSimulator object (using parameter constructor).
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)

    def test_MarketSimulator_Compute_BlackScholes_Calculation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to compute Black Scholes calculations.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        results = market_simulator.computeBlackScholesCall(
            current_stock_price=np.array([market_simulator.parameters.start_stock_price]),
            current_volatility=market_simulator.parameters.volatility
        )
        self.assertIsNotNone(results.price, msg=error_msg)
        self.assertIsNotNone(results.delta, msg=error_msg)
        print(f"Sample price:\n{results.price[:5]}")
        print(f"Sample delta:\n{results.delta[:5]}")

    def test_MarketSimulator_Compute_GBM_Simulation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to compute GBM simulations.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"

        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        gbm_stock_paths = market_simulator.computeGBMSimulation()
        self.assertIsNotNone(market_simulator, msg=error_msg)
        print(f"gbm_stock_paths:\n{gbm_stock_paths[:5, :10]}")

    def test_MarketSimulator_Compute_SABR_Simulation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to compute SABR simulations.
        :return:
        """
        n_sample_rows, n_sample_columns = 5, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        sabr_results = market_simulator.computeSABRSimulation()
        self.assertIsNotNone(sabr_results.underlying_price, msg=error_msg)
        self.assertIsNotNone(sabr_results.stochastic_volatility, msg=error_msg)
        actual_sabr_stock_paths = sabr_results.underlying_price[:n_sample_rows, :n_sample_columns]
        actual_sabr_stochastic_vol_paths = sabr_results.stochastic_volatility[:n_sample_rows, :n_sample_columns]
        print(f"sabr_stock_paths:\n{actual_sabr_stock_paths}\n\n")
        print(f"sabr_stochastic_vol_paths:\n{actual_sabr_stochastic_vol_paths}")

    def test_MarketSimulator_Compute_Heston_Simulation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to compute Heston model simulations.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        heston_results = market_simulator.runHestonSimulationUsingJax()
        self.assertIsNotNone(heston_results.stock_paths, msg=error_msg)
        self.assertIsNotNone(heston_results.volatility_paths, msg=error_msg)
        self.assertIsNotNone(heston_results.option_price_paths, msg=error_msg)
        self.assertIsNotNone(heston_results.option_deltas, msg=error_msg)
        print(f"result.stock_paths.shape: {heston_results.stock_paths.shape}")
        print(f"result.volatility_paths.shape: {heston_results.volatility_paths.shape}")
        print(f"result.option_price_paths.shape: {heston_results.option_price_paths.shape}")
        print(f"result.option_deltas.shape: {heston_results.option_deltas.shape}")

    def test_MarketSimulator_Compute_Implied_Volatility_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to compute implied volatility.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        sabr_results = market_simulator.computeSABRSimulation()
        self.assertIsNotNone(sabr_results.underlying_price, msg=error_msg)
        self.assertIsNotNone(sabr_results.stochastic_volatility, msg=error_msg)
        implied_vol = market_simulator.computeImpliedVolatility(
            sabr_results.underlying_price,
            sabr_results.stochastic_volatility)
        actual_implied_vol = implied_vol[:n_sample_rows, :n_sample_columns]
        print(f"Implied volatility:\n{actual_implied_vol.tolist()}")

    def test_MarketSimulator_Compute_Barlett_Delta_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to compute Barlett's Delta.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        sabr_results = market_simulator.computeSABRSimulation()
        self.assertIsNotNone(sabr_results.underlying_price, msg=error_msg)
        self.assertIsNotNone(sabr_results.stochastic_volatility, msg=error_msg)
        implied_vol = market_simulator.computeImpliedVolatility(
            sabr_results.underlying_price,
            sabr_results.stochastic_volatility)
        bartlett_delta = market_simulator.computeBartlettDelta(
            sabr_stock_price=sabr_results.underlying_price,
            implied_volatility=implied_vol
        )
        self.assertIsNotNone(bartlett_delta, msg=error_msg)
        actual_bartlett_delta = bartlett_delta[:n_sample_rows, :n_sample_columns]
        print(f"Bartlett delta:\n{bartlett_delta[:n_sample_rows, :n_sample_columns].tolist()}")

    def test_MarketSimulator_Run_Of_GBM_Simulation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to run the GBM simulation.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        results = market_simulator.runGBMSimulation()
        self.assertIsNotNone(results, msg=error_msg)
        self.assertIsNotNone(results.gbm_stock_paths, msg=error_msg)
        self.assertIsNotNone(results.gbm_call_price, msg=error_msg)
        self.assertIsNotNone(results.gbm_delta, msg=error_msg)
        print(f"results.gbm_stock_paths.shape: {results.gbm_stock_paths.shape}")
        print(f"Top 10 gbm_stock_paths: {results.gbm_stock_paths[10,:10]}")
        print(f"Bottom 10 gbm_stock_paths: {results.gbm_stock_paths[10, -10:]}")

        actual_gbm_sock_paths = results.gbm_stock_paths[:n_sample_rows, :n_sample_columns]
        actual_gbm_call_price = results.gbm_call_price[:n_sample_rows, :n_sample_columns]
        actual_gbm_delta = results.gbm_delta[:n_sample_rows, :n_sample_columns]
        print(f"gbm_sock_paths:\n{results.gbm_stock_paths[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"gbm_option_price:\n{results.gbm_call_price[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"gbm_option_delta:\n{results.gbm_delta[:n_sample_rows, :n_sample_columns].tolist()}")

    def test_MarketSimulator_Run_Of_SABR_Simulation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to run the SABR simulation.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        results = market_simulator.runSABRSimulation()
        self.assertIsNotNone(results, msg=error_msg)
        self.assertIsNotNone(results.sabr_stock_price, msg=error_msg)
        self.assertIsNotNone(results.sabr_call_price, msg=error_msg)
        self.assertIsNotNone(results.sabr_delta, msg=error_msg)

        self.assertIsNotNone(results.sabr_volatility, msg=error_msg)
        self.assertIsNotNone(results.sabr_implied_volatility, msg=error_msg)
        self.assertIsNotNone(results.sabr_bartlett_delta, msg=error_msg)

        actual_sabr_sock_paths = results.sabr_stock_price[:n_sample_rows, :n_sample_columns]
        actual_sabr_call_price = results.sabr_call_price[:n_sample_rows, :n_sample_columns]
        actual_sabr_delta = results.sabr_delta[:n_sample_rows, :n_sample_columns]

        actual_sabr_volatility = results.sabr_volatility[:n_sample_rows, :n_sample_columns]
        actual_sabr_implied_volatility = results.sabr_implied_volatility[:n_sample_rows, :n_sample_columns]
        actual_sabr_bartlett_delta = results.sabr_bartlett_delta[:n_sample_rows, :n_sample_columns]

        print(f"sabr_sock_paths:\n{results.sabr_stock_price[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"sabr_option_price:\n{results.sabr_call_price[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"sabr_option_delta:\n{results.sabr_delta[:n_sample_rows, :n_sample_columns].tolist()}")

        print(f"sabr_volatility:\n{results.sabr_volatility[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"sabr_implied_volatility:\n{results.sabr_implied_volatility[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"sabr_bartlett_delta:\n{results.sabr_bartlett_delta[:n_sample_rows, :n_sample_columns].tolist()}")

    def test_MarketSimulator_Run_Of_Heston_Simulation_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to run the Heston model Monte Carlo simulation.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        simulation_results = market_simulator.runHestonSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        print(f"result.stock_paths.shape: {simulation_results.stock_paths.shape}")
        print(f"result.volatility_paths.shape: {simulation_results.volatility_paths.shape}")
        print(f"result.option_price_paths.shape: {simulation_results.option_price_paths.shape}")

    def test_MarketSimulator_Calculation_Of_GBM_hedging_Strategy_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the GBM Hedging strategy.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        simulation_results = market_simulator.runGBMSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.gbm_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        actual_black_scholes_holding = hedging_results.holding_black_scholes[:n_sample_rows, :n_sample_columns]
        actual_black_scholes_trading = hedging_results.trading_black_scholes[:n_sample_rows, :n_sample_columns]
        print(f"black_scholes_holding:\n{actual_black_scholes_holding.tolist()}")
        print(f"black_scholes_trading:\n{actual_black_scholes_trading.tolist()}")

    def test_MarketSimulator_Calculation_Of_SABR_hedging_Strategy_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the SABR Hedging strategy.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)

        simulation_results = market_simulator.runSABRSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_stock_price, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_call_price, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_delta, msg=error_msg)

        self.assertIsNotNone(simulation_results.sabr_volatility, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_implied_volatility, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_bartlett_delta, msg=error_msg)

        hedging_results = market_simulator.computeHedgingStrategy(
            method="SABR",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.sabr_delta,
            bartlett_delta=simulation_results.sabr_bartlett_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_bartlett, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_bartlett, msg=error_msg)

        actual_black_scholes_holding = hedging_results.holding_black_scholes[:n_sample_rows, :n_sample_columns]
        actual_black_scholes_trading = hedging_results.trading_black_scholes[:n_sample_rows, :n_sample_columns]
        actual_bartlett_holding = hedging_results.holding_bartlett[:n_sample_rows, :n_sample_columns]
        actual_bartlett_trading = hedging_results.trading_bartlett[:n_sample_rows, :n_sample_columns]

        print(f"Black_scholes_holding:\n{hedging_results.holding_black_scholes[:n_sample_rows, :n_sample_columns].tolist()}")
        print(f"Black_scholes_trading:\n{hedging_results.trading_black_scholes[:n_sample_rows, :n_sample_columns].tolist()}")
        print(
            f"Bartlett_holding:\n{hedging_results.holding_bartlett[:n_sample_rows, :n_sample_columns].tolist()}")
        print(
            f"Bartlett_trading:\n{hedging_results.trading_bartlett[:n_sample_rows, :n_sample_columns].tolist()}")

    def test_MarketSimulator_Calculation_Black_Scholes_Adjusted_Pnl_Process_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the Black Scholes (delta hedging)
        Adjusted PnL process.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        simulation_results = market_simulator.runGBMSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.gbm_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        apl_results = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=simulation_results.gbm_stock_paths,
            option_price=simulation_results.gbm_call_price * market_simulator.parameters.notional,
            holding=hedging_results.holding_black_scholes
        )
        self.assertIsNotNone(apl_results, msg=error_msg)
        self.assertIsNotNone(apl_results.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results.accounting_pnl, msg=error_msg)
        actual_black_scholes_apl = apl_results.accounting_pnl[:n_sample_rows, :n_sample_columns]
        actual_black_scholes_holding_lagged = apl_results.holding_lagged[:n_sample_rows, :n_sample_columns]
        print(f"Black scholes APL:\n{actual_black_scholes_apl}")
        print(f"Black scholes holding_lagged:\n{actual_black_scholes_holding_lagged}")

    def test_MarketSimulator_Calculation_SABR_Adjusted_Pnl_Process_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the SABR (Bartlett hedging)
        Adjusted PnL process.
        :return:
        """
        n_sample_rows, n_sample_columns = 2, 10
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        simulation_results = market_simulator.runSABRSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_stock_price, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_call_price, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_delta, msg=error_msg)

        self.assertIsNotNone(simulation_results.sabr_volatility, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_implied_volatility, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_bartlett_delta, msg=error_msg)


        hedging_results = market_simulator.computeHedgingStrategy(
            method="SABR",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.sabr_delta,
            bartlett_delta=simulation_results.sabr_bartlett_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_bartlett, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_bartlett, msg=error_msg)

        apl_results = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=simulation_results.sabr_stock_price,
            option_price=simulation_results.sabr_call_price * market_simulator.parameters.notional,
            holding=hedging_results.holding_bartlett
        )
        self.assertIsNotNone(apl_results, msg=error_msg)
        self.assertIsNotNone(apl_results.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results.accounting_pnl, msg=error_msg)
        actual_bartlett_apl = apl_results.accounting_pnl[:n_sample_rows, :n_sample_columns]
        actual_bartlett_holding_lagged = apl_results.holding_lagged[:n_sample_rows, :n_sample_columns]
        print(f"Bartlett APL:\n{actual_bartlett_apl}")
        print(f"Bartlett holding_lagged:\n{actual_bartlett_holding_lagged}")

    def test_MarketSimulator_Calculation_Classical_Hedging_For_Black_Scholes_Approach_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the classical hedging
        Black Scholes (delta hedging)
        :return:
        """
        n_sample_rows = 2
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        simulation_results = market_simulator.runGBMSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.gbm_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        apl_results = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=simulation_results.gbm_stock_paths,
            option_price=simulation_results.gbm_call_price * market_simulator.parameters.notional,
            holding=hedging_results.holding_black_scholes
        )
        self.assertIsNotNone(apl_results, msg=error_msg)
        self.assertIsNotNone(apl_results.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results.accounting_pnl, msg=error_msg)
        classical_hedging_results = market_simulator.evaluateAgainstClassicalHedging(
            accounting_pnl=apl_results.accounting_pnl,
            option_price=simulation_results.gbm_call_price
        )
        actual_evaluation_function = np.mean(classical_hedging_results.evaluation_function)
        actual_percentage_mean_ratio = classical_hedging_results.percentage_mean_ratio
        actual_percentage_std_ratio = classical_hedging_results.percentage_std_ratio

        print(f"Black scholes Y(0):\n{actual_evaluation_function}")
        print(f"Black scholes percentageMeanRatio:\n{actual_percentage_mean_ratio}")
        print(f"Black scholes percentageStdRatio:\n{actual_percentage_std_ratio}")

    def test_MarketSimulator_Calculation_Classical_Hedging_For_Bartlett_Approach_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the classical hedging
        (Bartlett hedging)
        :return:
        """
        n_sample_rows = 2
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        simulation_results = market_simulator.runSABRSimulation()
        self.assertIsNotNone(simulation_results, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_stock_price, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_call_price, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_delta, msg=error_msg)

        self.assertIsNotNone(simulation_results.sabr_volatility, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_implied_volatility, msg=error_msg)
        self.assertIsNotNone(simulation_results.sabr_bartlett_delta, msg=error_msg)

        hedging_results = market_simulator.computeHedgingStrategy(
            method="SABR",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.sabr_delta,
            bartlett_delta=simulation_results.sabr_bartlett_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_bartlett, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_bartlett, msg=error_msg)

        apl_results_sabr = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=simulation_results.sabr_stock_price,
            option_price=simulation_results.sabr_call_price * market_simulator.parameters.notional,
            holding=hedging_results.holding_black_scholes
        )

        apl_results_sabr_bartlett = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=simulation_results.sabr_stock_price,
            option_price=simulation_results.sabr_call_price * market_simulator.parameters.notional,
            holding=hedging_results.holding_bartlett
        )
        self.assertIsNotNone(apl_results_sabr, msg=error_msg)
        self.assertIsNotNone(apl_results_sabr.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results_sabr.accounting_pnl, msg=error_msg)

        self.assertIsNotNone(apl_results_sabr_bartlett, msg=error_msg)
        self.assertIsNotNone(apl_results_sabr_bartlett.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results_sabr_bartlett.accounting_pnl, msg=error_msg)

        sabr_hedging_results = market_simulator.evaluateAgainstClassicalHedging(
            accounting_pnl=apl_results_sabr.accounting_pnl,
            option_price=simulation_results.sabr_call_price
        )

        sabr_bartlett_hedging_results = market_simulator.evaluateAgainstClassicalHedging(
            accounting_pnl=apl_results_sabr_bartlett.accounting_pnl,
            option_price=simulation_results.sabr_call_price
        )
        actual_evaluation_function_sabr = np.mean(sabr_hedging_results.evaluation_function)
        actual_percentage_mean_ratio_sabr = sabr_hedging_results.percentage_mean_ratio
        actual_percentage_std_ratio_sabr = sabr_hedging_results.percentage_std_ratio

        actual_evaluation_function_sabr_bartlett = np.mean(sabr_bartlett_hedging_results.evaluation_function)
        actual_percentage_mean_ratio_sabr_bartlett = sabr_bartlett_hedging_results.percentage_mean_ratio
        actual_percentage_std_ratio_sabr_bartlett = sabr_bartlett_hedging_results.percentage_std_ratio

        print(f"SABR Y(0):\n{actual_evaluation_function_sabr}")
        print(f"SABR percentageMeanRatio:\n{actual_percentage_mean_ratio_sabr}")
        print(f"SABR percentageStdRatio:\n{actual_percentage_std_ratio_sabr}")
        print(self.line_divider)
        print(f"Bartlett Y(0):\n{actual_evaluation_function_sabr_bartlett}")
        print(f"Bartlett percentageMeanRatio:\n{actual_percentage_mean_ratio_sabr_bartlett}")
        print(f"Bartlett percentageStdRatio:\n{actual_percentage_std_ratio_sabr_bartlett}")

    def test_MarketSimulator_Calculation_Classical_Hedging_For_Black_Scholes_Analysis_Is_Valid(self):
        """
        Test the validity of using MarketSimulator object to calculate the classical hedging
        Black Scholes (delta hedging analysis)
        :return:
        """
        n_sample_rows = 2
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(market_simulator, msg=error_msg)
        simulation_results = market_simulator.runGBMSimulation()
        expected_option_price = np.array([3.90725713, 4.13392309, 5.48813047, 5.54383339, 5.0754437])
        expected_option_delta = np.array([0.51953629, 0.54445129, 0.64449637, 0.65699042, 0.63738316])
        print(self.line_divider)
        print("\nActual results (First 5 option prices for 1st path):")
        print(f"option prices: {simulation_results.gbm_call_price[0,:5]}")
        print(f"delta prices: {simulation_results.gbm_delta[0, :5]}")

        print("\nExpected results First 5 option prices for 1st path):")
        print(f"option prices: {expected_option_price}")
        print(f"delta prices: {expected_option_delta}")
        print(self.line_divider)
        # np.testing.assert_array_almost_equal(actual_gbm_stock_paths, desired_gbm_stock_paths, decimal=6, err_msg=error_msg)
        self.assertIsNotNone(simulation_results, msg=error_msg)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=simulation_results.gbm_delta
        )
        self.assertIsNotNone(hedging_results, msg=error_msg)
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)
        apl_results = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=simulation_results.gbm_stock_paths,
            option_price=simulation_results.gbm_call_price * market_simulator.parameters.notional,
            holding=hedging_results.holding_black_scholes
        )
        self.assertIsNotNone(apl_results, msg=error_msg)
        self.assertIsNotNone(apl_results.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results.accounting_pnl, msg=error_msg)
        classical_hedging_results = market_simulator.evaluateAgainstClassicalHedging(
            accounting_pnl=apl_results.accounting_pnl,
            option_price=simulation_results.gbm_call_price
        )
        actual_evaluation_function = np.mean(classical_hedging_results.evaluation_function)
        actual_percentage_mean_ratio = classical_hedging_results.percentage_mean_ratio
        actual_percentage_std_ratio = classical_hedging_results.percentage_std_ratio

        print(f"Black scholes Y:\n{actual_evaluation_function}")
        print(f"Black scholes percentageMeanRatio:\n{actual_percentage_mean_ratio}")
        print(f"Black scholes percentageStdRatio:\n{actual_percentage_std_ratio}")



if __name__ == '__main__':
    ut.main()
