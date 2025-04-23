from jaxfin.price_engine.black_scholes import european_price, delta_european
from jax import vmap
import numpy as np
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# from black_scholes_env_cont import BlackScholesEnvCont

v_delta_european = vmap(delta_european, in_axes=(0, None, None, None, None))



def flatten(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs).flatten()

    return wrapper

def observations2Dict(
        obs: np.ndarray,
        is_legacy: bool = False,
):
    if is_legacy:
        return {
            "holdings": obs[0],
            "moneyness": obs[1],
            "time_to_expiration": obs[2],
        }
    return {
        "log_price_strike": obs[0],
        "vol": obs[1],
        "time_to_expiration": obs[2],
        "bs_delta": obs[3],
        "normalized_call_price": obs[4],
        "hedging_delta": obs[5],
    }

# def makeEnv(s0, strike, expiry, r, mu, vol, n_steps):
#     def _create_env():
#         return BlackScholesEnvCont(s0, strike, expiry, r, mu, vol, n_steps)
#     return _create_env
#
# def makeVecEnv(s0: float, strike: float, expiry:float, r: float, mu: float, vol: float, n_steps: int) -> Env:
#     vec_env = DummyVecEnv([makeEnv(s0, strike, expiry, r, mu, vol, n_steps) for _ in range(5)])
#     return VecMonitor(vec_env, info_keywords=("current_pnl",), filename='./training.log')