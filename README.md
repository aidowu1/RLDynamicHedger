# RLDynamicHedger
RLDynamicHedger is a Reinforcement Learning based dynamic hedging solution for European Call options


## Abstract
Reinforcement Learning (RL) is considered one of the 3 sub-fields in Machine Learning (ML) taxonomy, with the other 2 sub-fields being Supervised and Unsupervised Learning. Lately, RL is increasingly being used to solve financial problems which require sequential decision making in uncertain environments (financial markets), these include asset allocation, optimal order execution, derivative pricing/hedging and algo trading to name a few. In this work, I will develop and review the application of some key RL models for hedging derivatives in the presence of market frictions such as transaction costs. For brevity, data used in this project will be generated synthetically using Geometric Brownian Motion Monte Carlo simulation of paths for constant volatility modelling. Additionally, stochastic volatility modelling experiments will be facilitated by SABR and Heston simulation paths.  Fundamentally, the RL models will be used to optimize the hedging strategy for maximum reward and minimum risk. This work will compare the performance of the selected RL models against the classical standard approach of delta hedging (baseline). 


## Instructions on how to run RLDynamicHedger solution

### Introduction
RLDynamicHedger is a solution developed in python for demonstrating the use of Reinforcement Learning agents to dynamically hedger European Call options (i.e. it can be extended for Put options and other option types). It utilizes several open-source python libraries such as: Pandas, Numpy, Pytorch, Gymnasium, Stable-baselines3, Tensorboard to name a few (for the full list of python packages, please have a look at the requirements.txt in the code repo)

RLDynamicHedger has been developed as a library (i.e. as a project in Pycharm, although this equally be developed in other IDEs such as VS Code). For demonstration purposes, JupyterLab (notebooks) have been developed to act as entry-point frontends to demonstrate various aspects of RLDynamicHedger functionality.

Also to complement the JupyterLab notebooks, python scripts have also been provided to allow the user to run the RLDynamicHedger via command line interface.

### Installation of RLDynamicHedger
To install the application please follow these steps:
 - Clone the code Git repository:
   - Launch your local version of command prompt (cmd)
   - Ensure you have a python distribution on your PC, for this solution I used the Miniconda distribution. Please following  the instruction described here to install minconda on your PC.
   - Create a python virtual environment for this solution following these steps:
     - conda create -n hedging_env python=3.10 -y
   - Activate the newly created environment:
     - conda activate hedging_env		
   - Clone the code repo by typing the following in the CMD prompt:
     - git clone git@github.com:aidowu1/RLDynamicHedger.git 
   - Change your working directory to the root folder of the application:
     - cd RLDynamicHedger-Final
   - Using the PIP package manager, install the python packages required to run the solution:
     - pip intall -r hedging_env_3_requirements.txt
   - Launch Jupyterlab notebook:
     - jupyter lab
   - Browse to the notebooks directory and run your desired demo noetbook

### RLDynamicHedger High-level Code structure
The high-level structure of the code is as follows:
 - RLDynamicHedger-Final (Project folder)
   - src (Library source code sub-folder)
   - notebooks (Jupyterlab notebooks sub-folder, used to demo the library functionality)
   - scripts (Python scripts sub-folder that can be used to alternatively demo the functionality via command-line arguments)
   - data (folder contains the GBM, SABR and Heston simulation data for the experiment use cases) 
   - plots (folder for storing the generated results plots)
   - tensorboard (folder to store Tensorboard run metrics and traces)
   - logs (Logs sub-folder)
   - model (Archived trained/tuned RL models sub-folder)
   - hedging_env_3_requirements.txt (List of python packages)

### Source code folder structure 
The source code folder is structured as follows:
 - main (Sub-folder of the library source code):
   - environment (Sub-folder contains code to create the RL)
   - market_simulator (Sub-folder contains code to simulate the RL environment)
   - performance_metric (sub-folder contains code for the RL performance metrics)
   - rl_algorithms (Sub-folder contains code of the RL algorithms used for this solution):
     - hyper_parameter (Code for hyper-parameter tuning)
     - policy_gradient (Code for policy gradient RL algorithms)
     - train_evaluate_test (Code to train, evaluate and test the RL solution) 
   - settings (Sub-folder used to specify the JSON configuration settings of the RL solution)
   - utility (sub-folder used to specify utility functions used in the RL solution)
   - test (sub-folder of the library source code unit/integration tests)

### RLDynamicHedger Notebooks
There are number of Jupyterlab notebooks used to demo various aspects of the RL solution, these include:
 - Demos hyper-parameter tuning/training of the RL models for each simulation use cases (GBM, SABR and Heston) and RL algorithms (TD3, DDPG, SAC and PPO):
   - RLDynamicHedger-Hyper-parameter-Tuning-All-RL-Algorithms.ipynb
 - Demos the reading of the tuned hyper-parameters for each use-case and RL algorithm:
   - RLDynamicHedger - Reads-Optimal-Hyper-parameter-tuned RL agents.ipynb
 - Demos the calculation and plotting of the Mean Normalized Error (MNE) and Standard Error of MNE (SEM)  comparative performance metrics of the RL agents:
   - RLDynamicHedger-Comparative-RL-Performance-Metrics.ipynb
 - Demos the plotting of reward curves for all the RL algorithms (for high expiry time GBM simulation use case):
   - RLDynamicHedger-Reward-Curves-For-Trained-RL-Models.ipynb
 - Demos 4 RL models inference hedging performance results for any of the 8 experiment settings and all simulation (GBM, SABR and Heston) use cases:
   - RLDynamicHedger-Evaluation-For-Tuned-RL-Algo – V2.ipynb
 - Demos hedging performance results (PnL, Rewards, Delta and Trading Cost distributions) and tables of performance metrics for all the 4 RL algorithms and per simulation use case. The computed use cases include:
        - High trading frequency
        - High trading cost
        - High moneyness (S/K = 1.1)
        - High expiry time
        - Low trading frequency
        - Low trading cost
        - Low expiry time
        - Low moneyness (S/K = 0.8)
   - RLDynamicHedger-Generate-Results-All-Models - V2.ipynb
 - Demos the consolidation of all the out-of-sample metrics for all the 8 experiment use cases
   - RLDynamicHedger-Consolidate-Hull-Metrics-Per-Use-Case.ipynb
 - Python module that has a helper classes used to set the experiment settings/parameters for each of the 8 use cases:
   - experiment_use_case.py

### RLDynamicHedger Scripts
Several python scripts can be used to alternatively demo some of the key features of the RL hedger solution, these include:
 - Demos the analysis of the RL hedging environment and the agents:
   - analysis_of_hedging_env.py
 - Demos hyper-parameter tuning/training of the RL models for each simulation use cases (GBM, SABR and Heston) and RL algorithms (TD3, DDPG, SAC and PPO):
   - tune_hedger_rl_model.py
 - Demos the plotting of the Mean Normalized Error (MNE) and Standard Error of MNE (SEM) for all the RL algorithms:
   - generate_hedger_all_hull_metrics_results.py
 - Demos the plotting of reward curves for all the RL algorithms (for high expiry time GBM simulation use case):
   - generate_rl_reward_plots.py
 - Demos the generation of mean and standard deviation of trading cost results for TD3 RL algorithm for special use cases:
   - generate_hedger_rl_results_single_model_special_case.py
 - Demos the calculation of comparative performance metrics of the RL agents:
   - generate_hedger_rl_comparative_perf_results.py
 - Demos the persistence of a trained RL model:
   - train_save_hedger_rl_model.py
 - Demos hedging performance results (PnL, Rewards, Delta and Trading Cost distributions) performance metrics for all the 4 RL algorithms and per simulation use case:
   - generate_hedger_rl_model_results_all_models.py
   - run_generate_all_rl_agent_density_plots.bat


