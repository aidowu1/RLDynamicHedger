# RLDynamicHedger
RLDynamicHedger is a Reinforcement Learning based dynamic hedging solution for European Call options


## Abstract
Reinforcement Learning (RL) is considered one of the 3 sub-fields in Machine Learning (ML) taxonomy, with the other 2 sub-fields being Supervised and Unsupervised Learning. Lately, RL is increasingly being used to solve financial problems which require sequential decision making in uncertain environments (financial markets), these include asset allocation, optimal order execution, derivative pricing/hedging and algo trading to name a few. In this work, I will develop and review the application of some key RL models for hedging derivatives in the presence of market frictions such as transaction costs. For brevity, data used in this project will be generated synthetically using Geometric Brownian Motion Monte Carlo simulation of paths for constant volatility modelling. Additionally, stochastic volatility modelling experiments will be facilitated by SABR and Heston simulation paths.  Fundamentally, the RL models will be used to optimize the hedging strategy for maximum reward and minimum risk. This work will compare the performance of the selected RL models against the classical standard approach of delta hedging (baseline). 

