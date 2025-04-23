echo "Test script used to generate hedger RL kernel distribution estimation plots"

python generate_hedger_rl_model_results_all_models.py --hedging_type gbm --is_recompute False

python generate_hedger_rl_model_results_all_models.py --hedging_type sabr --is_recompute False

python generate_hedger_rl_model_results_all_models.py --hedging_type heston --is_recompute False