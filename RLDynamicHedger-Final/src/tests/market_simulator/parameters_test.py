import unittest as ut
import pathlib as p
import os
import inspect
from pprint import pprint

import src.main.configs_global as configs
from src.main.utility.settings_reader import SettingsReader
from src.main.market_simulator.parameters import Parameters
from src.main.utility.utils import Helpers

class ParametersTest(ut.TestCase):
    """
    Test suit for the 'Parameters' class.
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        :return: None
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

    def test_Parameters_Construction_Is_Valid(self):
        """
        Test the validity of construction of the Parameters object.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        settings = SettingsReader(configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(settings, msg=error_msg)
        file_exists = settings.file_exists
        self.assertTrue(file_exists, msg=error_msg)
        settings_data = settings.read()
        self.assertIsNotNone(settings_data, msg=error_msg)
        print(f"Setting data is:")
        pprint(settings_data)
        parameters = Parameters(**settings_data)
        self.assertIsNotNone(parameters, msg=error_msg)
        print(f"Parameters data is:")
        pprint(parameters)





if __name__ == '__main__':
    ut.main()
