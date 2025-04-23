import unittest as ut
import inspect
import os
import pathlib as p

import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.market_simulator.generate_simulations import SimulationGenerator

class SimulationGeneratorTest(ut.TestCase):
    """
    Integration test suit for "SimulationGenerator" component
    """
    def setUp(self):
        """
        Test set-up fixture
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.parameter_settings_filename = configs.DEFAULT_SETTINGS_NAME

    def test_SimulationGenerator_Constructor_Is_Valid(self):
        """
        Test the validity of "SimulationGenerator" contructor
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulator = SimulationGenerator(parameter_settings_filename=self.parameter_settings_filename)
        self.assertIsNotNone(simulator, msg=error_msg)

    def test_SimulationGenerator_Generate_GBM_Simulation_Is_Valid(self):
        """
        Test the validity of "SimulationGenerator" GBM simulation
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulator = SimulationGenerator(parameter_settings_filename=self.parameter_settings_filename)
        self.assertIsNotNone(simulator, msg=error_msg)
        status = simulator.generateGbmSimulation()
        self.assertTrue(status, msg=error_msg)

    def test_SimulationGenerator_Generate_SABR_Simulation_Is_Valid(self):
        """
        Test the validity of "SimulationGenerator" SABR simulation
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulator = SimulationGenerator(parameter_settings_filename=self.parameter_settings_filename)
        self.assertIsNotNone(simulator, msg=error_msg)
        status = simulator.generateSabrSimulation()
        self.assertTrue(status, msg=error_msg)

    def test_SimulationGenerator_Generate_Heston_Simulation_Is_Valid(self):
        """
        Test the validity of "SimulationGenerator" Heston simulation
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        simulator = SimulationGenerator(parameter_settings_filename=self.parameter_settings_filename)
        self.assertIsNotNone(simulator, msg=error_msg)
        status = simulator.generateHestonSimulation()
        self.assertTrue(status, msg=error_msg)


if __name__ == '__main__':
    ut.main()
