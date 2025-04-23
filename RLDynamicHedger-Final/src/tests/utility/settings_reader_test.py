import unittest as ut
import pathlib as p
import os
import inspect
from pprint import pprint

import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.utility.settings_reader import SettingsReader

class SettingsReaderTest(ut.TestCase):
    """
    Test suit for the 'SettingsReader' class.
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        :return: None
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

    def test_SettingsReader_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the SettingsReader object.
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        settings = SettingsReader(configs.DEFAULT_SETTINGS_NAME)
        self.assertIsNotNone(settings, msg=error_msg)
        file_exists = settings.file_exists
        self.assertTrue(file_exists, msg=error_msg)

    def test_SettingsReader_Deserialize_Data_Is_Valid(self):
        """
        Test the validity of the deserialization of JSON data using SettingsReader object.
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



if __name__ == '__main__':
    ut.main()
