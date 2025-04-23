import unittest as ut
import inspect
import numpy as np
import os

from src.main.market_simulator.simulator import MarketSimulator
from src.main.market_simulator.caching import SimulationDataCache
import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.tests.third_party.simulation_settings import SimulationSettings
from src.tests.third_party.simulation import (simulateGBM, simulateSABR,
                                              evaluate, APL_process, hedgingStrategy)

class HedgingPerformanceMetricsIntegrationTest(ut.TestCase):
    def setUp(self):
        """
        Test set-up fixture
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.line_divider = "=====" * 10
        self.simulation_settings = SimulationSettings()

    def test_Third_Party_Hedging_Performance_Calculations_GBM_Paths(self):
        """
        Test the validity of the third party hedging performance calculations
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        stock_price_gbm, option_price_gbm, delta_gbm = simulateGBM(
            self.simulation_settings.n,
            self.simulation_settings.T,
            self.simulation_settings.dt,
            self.simulation_settings.S0,
            self.simulation_settings.mu,
            self.simulation_settings.r,
            self.simulation_settings.q,
            self.simulation_settings.sigma,
            self.simulation_settings.days,
            self.simulation_settings.freq,
            self.simulation_settings.K
        )
        self.assertIsNotNone(stock_price_gbm, msg=error_msg)
        self.assertIsNotNone(option_price_gbm, msg=error_msg)
        self.assertIsNotNone(delta_gbm, msg=error_msg)

    def test_Third_Party_Hedging_Performance_Calculations_Hedging_Strategy(self):
        """
        Test the validity of the third party hedging strategy calculations
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        stock_price_gbm, option_price_gbm, delta_gbm = simulateGBM(
            self.simulation_settings.n,
            self.simulation_settings.T,
            self.simulation_settings.dt,
            self.simulation_settings.S0,
            self.simulation_settings.mu,
            self.simulation_settings.r,
            self.simulation_settings.q,
            self.simulation_settings.sigma,
            self.simulation_settings.days,
            self.simulation_settings.freq,
            self.simulation_settings.K
        )
        self.assertIsNotNone(stock_price_gbm, msg=error_msg)
        self.assertIsNotNone(option_price_gbm, msg=error_msg)
        self.assertIsNotNone(delta_gbm, msg=error_msg)

        simulation_type = "GBM"
        bl_delta = np.array([[0]])
        trading_gbm, holding_gbm = hedgingStrategy(
            simulation_type,
            self.simulation_settings.notional,
            delta_gbm, bl_delta)

        self.assertIsNotNone(trading_gbm, msg=error_msg)
        self.assertIsNotNone(holding_gbm, msg=error_msg)

        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=delta_gbm
        )
        self.assertIsNotNone(hedging_results.holding_black_scholes, msg=error_msg)
        self.assertIsNotNone(hedging_results.trading_black_scholes, msg=error_msg)

        np.testing.assert_array_almost_equal(
            holding_gbm,
            hedging_results.holding_black_scholes,
            decimal=6,
            err_msg=error_msg
        )

        np.testing.assert_array_almost_equal(
            trading_gbm,
            hedging_results.trading_black_scholes,
            decimal=6,
            err_msg=error_msg
        )

    def test_Third_Party_Hedging_Performance_Calculations_APL_process(self):
        """
        Test the validity of the third party hedging strategy Accounting PnL (APL) process calculations
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        stock_price_gbm, option_price_gbm, delta_gbm = simulateGBM(
            self.simulation_settings.n,
            self.simulation_settings.T,
            self.simulation_settings.dt,
            self.simulation_settings.S0,
            self.simulation_settings.mu,
            self.simulation_settings.r,
            self.simulation_settings.q,
            self.simulation_settings.sigma,
            self.simulation_settings.days,
            self.simulation_settings.freq,
            self.simulation_settings.K
        )

        simulation_type = "GBM"
        bl_delta = np.array([[0]])
        trading_gbm, holding_gbm = hedgingStrategy(
            simulation_type,
            self.simulation_settings.notional,
            delta_gbm, bl_delta)

        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=delta_gbm
        )

        APL_gbm, holding_lagged_gbm = APL_process(
            stock_price_gbm,
            option_price_gbm * self.simulation_settings.notional,
            holding_gbm,
            self.simulation_settings.K,
            self.simulation_settings.notional,
            self.simulation_settings.kappa
        )

        self.assertIsNotNone(APL_gbm, msg=error_msg)
        self.assertIsNotNone(holding_lagged_gbm, msg=error_msg)

        apl_results = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=stock_price_gbm,
            option_price=option_price_gbm * market_simulator.parameters.notional,
            holding=hedging_results.holding_black_scholes
        )

        self.assertIsNotNone(apl_results.holding_lagged, msg=error_msg)
        self.assertIsNotNone(apl_results.accounting_pnl, msg=error_msg)

        np.testing.assert_array_almost_equal(
            holding_lagged_gbm,
            apl_results.holding_lagged,
            decimal=6,
            err_msg=error_msg
        )

        np.testing.assert_array_almost_equal(
            APL_gbm,
            apl_results.accounting_pnl,
            decimal=6,
            err_msg=error_msg
        )

    def test_Third_Party_Hedging_Performance_Calculations_Evaluate_Y(self):
        """
        Test the validity of the third party hedging performance Y(0) calculations
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        stock_price_gbm, option_price_gbm, delta_gbm = simulateGBM(
            self.simulation_settings.n,
            self.simulation_settings.T,
            self.simulation_settings.dt,
            self.simulation_settings.S0,
            self.simulation_settings.mu,
            self.simulation_settings.r,
            self.simulation_settings.q,
            self.simulation_settings.sigma,
            self.simulation_settings.days,
            self.simulation_settings.freq,
            self.simulation_settings.K
        )

        simulation_type = "GBM"
        bl_delta = np.array([[0]])
        trading_gbm, holding_gbm = hedgingStrategy(
            simulation_type,
            self.simulation_settings.notional,
            delta_gbm, bl_delta)

        market_simulator = MarketSimulator(parameter_settings_filename=configs.DEFAULT_SETTINGS_NAME)
        hedging_results = market_simulator.computeHedgingStrategy(
            method="GBM",
            notional=market_simulator.parameters.notional,
            delta=delta_gbm
        )

        APL_gbm, holding_lagged_gbm = APL_process(
            stock_price_gbm,
            option_price_gbm * self.simulation_settings.notional,
            holding_gbm,
            self.simulation_settings.K,
            self.simulation_settings.notional,
            self.simulation_settings.kappa
        )

        apl_results = market_simulator.computeNotionalAdjustedPnlProcess(
            underlying_price=stock_price_gbm,
            option_price=option_price_gbm * market_simulator.parameters.notional,
            holding=hedging_results.holding_black_scholes
        )

        Y_gbm, mPerc_gbm, stdPerc_gbm = evaluate(
            APL_gbm,
            option_price_gbm,
            self.simulation_settings.c,
            self.simulation_settings.notional
        )

        self.assertIsNotNone(Y_gbm, msg=error_msg)
        self.assertIsNotNone(mPerc_gbm, msg=error_msg)
        self.assertIsNotNone(stdPerc_gbm, msg=error_msg)

        classical_hedging_results = market_simulator.evaluateAgainstClassicalHedging(
            accounting_pnl=apl_results.accounting_pnl,
            option_price=option_price_gbm
        )

        self.assertIsNotNone(classical_hedging_results.evaluation_function, msg=error_msg)
        self.assertIsNotNone(classical_hedging_results.percentage_mean_ratio, msg=error_msg)
        self.assertIsNotNone(classical_hedging_results.percentage_std_ratio, msg=error_msg)

        np.testing.assert_array_almost_equal(
            Y_gbm,
            classical_hedging_results.evaluation_function,
            decimal=6,
            err_msg=error_msg
        )

        np.testing.assert_array_almost_equal(
            mPerc_gbm,
            classical_hedging_results.percentage_mean_ratio,
            decimal=6,
            err_msg=error_msg
        )

        np.testing.assert_array_almost_equal(
            stdPerc_gbm,
            classical_hedging_results.percentage_std_ratio,
            decimal=6,
            err_msg=error_msg
        )




if __name__ == '__main__':
    ut.main()
