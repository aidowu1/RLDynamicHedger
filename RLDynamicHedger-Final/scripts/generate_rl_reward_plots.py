import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys

ROOT_FOLDER = f"{os.path.dirname(os.path.abspath(__file__))}\.."
print(f"Root folder: {ROOT_FOLDER}")
sys.path.append(ROOT_FOLDER)

from src.main.utility.enum_types import RLAgorithmType
from src.main.utility.logging import Logger
import src.main.configs_rl as configs2

class GeneratePlots:
    """
    Generate the RL results plots
    """
    def __init__(
            self,
            is_plot_2_screen: bool = False,
    ):
        """
        Constructor
        """
        self._logger = Logger.getLogger()
        self._is_plot_2_screen = is_plot_2_screen
        self._td3_data_df = pd.read_csv(configs2.HYPER_PARAMETER_REWARD_CURVE_DATA_PATH.format(
            RLAgorithmType.td3.name
        ))
        self._ddpg_data_df = pd.read_csv(configs2.HYPER_PARAMETER_REWARD_CURVE_DATA_PATH.format(
            RLAgorithmType.ddpg.name
        ))
        self._ppo_data_df = pd.read_csv(configs2.HYPER_PARAMETER_REWARD_CURVE_DATA_PATH.format(
            RLAgorithmType.ppo.name
        ))
        self._sac_data_df = pd.read_csv(configs2.HYPER_PARAMETER_REWARD_CURVE_DATA_PATH.format(
            RLAgorithmType.sac.name
        ))
        self._plot_hyper_parameter_plots_all = configs2.HYPER_PARAMETER_REWARD_CURVE_PATH.format("all")

    def plotRLModelRewardCurves(self):
        """
        Plot reward curves for all RL models
        :return: None
        """
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(
            self._sac_data_df["time_steps"],
            self._sac_data_df["rewards"],
            label="Reward (SAC)",
            marker='o',
            color='blue')
        axes[0].set_title("RL Model Reward Curves")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].set_xlabel("Time steps")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            self._ppo_data_df["time_steps"],
            self._ppo_data_df["rewards"],
            label="Reward (PPO)",
            marker='o',
            color='green')
        axes[1].set_title("")
        axes[1].set_ylabel("Cumulative Reward")
        axes[1].set_xlabel("Time steps")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(
            self._ddpg_data_df["time_steps"],
            self._ddpg_data_df["rewards"],
            label="Reward (DDPG)",
            marker='o',
            color='red')
        axes[2].set_title("")
        axes[2].set_ylabel("Cumulative Reward")
        axes[2].set_xlabel("Time steps")
        axes[2].legend()
        axes[2].grid(True)

        axes[3].plot(
            self._td3_data_df["time_steps"],
            self._td3_data_df["rewards"],
            label="Reward (TD3)",
            marker='o',
            color='cyan')
        axes[3].set_title("")
        axes[3].set_ylabel("Cumulative Reward")
        axes[3].set_xlabel("Time steps")
        axes[3].legend()
        axes[3].grid(True)
        plt.savefig(self._plot_hyper_parameter_plots_all)
        if self._is_plot_2_screen:
            plt.show()
        self._logger.info("RL Model Reward Curves Generated Successfully...")



def main():
    """
    Main function
    :return:
    """
    generate_plots = GeneratePlots()
    generate_plots.plotRLModelRewardCurves()


if __name__ == "__main__":
    main()

