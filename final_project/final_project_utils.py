from __future__ import annotations

import os
import random
import shutil
import heapq
import pickle
import psutil

from typing import Union, Any, Optional

from typing import Optional, Union, Sequence

import numpy as np

import modin.pandas as pd
import modin.config as cfg


import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy.spatial import KDTree
import ta
import kagglehub

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import HerReplayBuffer


import heapq, random, numpy as np
from typing import Optional, Sequence
from stable_baselines3.common.buffers import ReplayBuffer
from sklearn.neighbors import KDTree  # pip install scikit-learn


# from sklearn.preprocessing import MinMaxScaler
import dask.array as da
from dask_ml.preprocessing import MinMaxScaler


from signal import Signals
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
import stumpy
from numba import cuda
import IPython.display


def print_version():
    print("LATEST!!!")


# def encode_datetime(dt):
#     if isinstance(dt, pd.Timestamp):
#         hour = dt.hour
#         minute = dt.minute
#         dayofweek = dt.dayofweek
#     else:
#         # If dt is a string, parse it
#         dt = pd.to_datetime(dt)
#         hour = dt.hour
#         minute = dt.minute
#         dayofweek = dt.dayofweek
#     # Cyclical encoding
#     hour_sin = np.sin(2 * np.pi * hour / 24)
#     hour_cos = np.cos(2 * np.pi * hour / 24)
#     minute_sin = np.sin(2 * np.pi * minute / 60)
#     minute_cos = np.cos(2 * np.pi * minute / 60)
#     dayofweek_sin = np.sin(2 * np.pi * dayofweek / 7)
#     dayofweek_cos = np.cos(2 * np.pi * dayofweek / 7)
#     return [hour_sin, hour_cos, minute_sin, minute_cos, dayofweek_sin, dayofweek_cos]


# class StockTradingEnv(gym.Env):
#     """
#     A minimal trading environment that handles ONE stock DataFrame only.
#     Action:  scalar in [-1, 1]  (‑1 = sell all, +1 = invest all cash).
#     Observation: price features + balance + shares held + net worth + max
#                  net worth + current step.
#     """

#     metadata = {"render_modes": ["human"]}

#     def __init__(
#         self, stock_data: pd.DataFrame, transaction_cost_percent: float = 0.005
#     ):
#         super().__init__()

#         if stock_data.empty:
#             raise ValueError("stock_data DataFrame is empty")
#         if "close" not in stock_data.columns:
#             raise ValueError("'close' column is required in stock_data")

#         self.stock_data = stock_data.reset_index(drop=True)
#         self.n_features = self.stock_data.shape[1]

#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
#         # self.action_space = spaces.Discrete(3)

#         # features + balance + shares + net worth + max net worth + step
#         self.obs_shape = self.n_features + 5
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
#         )

#         self.initial_balance = 1_000_000.0
#         self.transaction_cost_percent = transaction_cost_percent

#         self.reset()

#     # ──────────────────────────────────────────────────────────────────
#     # Core helpers
#     # ──────────────────────────────────────────────────────────────────
#     def _next_observation(self):
#         frame = np.zeros(self.obs_shape, dtype=np.float32)

#         # price features
#         frame[: self.n_features] = self.stock_data.iloc[self.current_step].values

#         # portfolio data
#         idx = self.n_features
#         frame[idx] = self.balance
#         idx += 1
#         frame[idx] = self.shares_held
#         idx += 1
#         frame[idx] = self.net_worth
#         idx += 1
#         frame[idx] = self.max_net_worth
#         idx += 1
#         frame[idx] = self.current_step

#         return frame

#     # ──────────────────────────────────────────────────────────────────
#     # Gym API
#     # ──────────────────────────────────────────────────────────────────
#     def reset(self, seed: int | None = None, options: dict | None = None):
#         super().reset(seed=seed)

#         self.balance = self.initial_balance
#         self.shares_held = 0
#         self.net_worth = self.initial_balance
#         self.max_net_worth = self.initial_balance

#         self.total_shares_sold = 0
#         self.total_sales_value = 0.0

#         self.current_step = 0
#         return self._next_observation(), {}

#     def step(self, action):
#         # ensure scalar
#         action = float(np.array(action).flatten()[0])

#         self.current_step += 1
#         done = self.current_step >= len(self.stock_data) - 1

#         current_price = self.stock_data.iloc[self.current_step]["close"]

#         if action > 0:  # buy
#             # if action == 1:
#             shares_to_buy = int(self.balance * action / current_price)
#             cost = shares_to_buy * current_price
#             fee = cost * self.transaction_cost_percent
#             self.balance -= cost + fee
#             self.shares_held += shares_to_buy

#         elif action < 0:  # sell
#             # elif action == 2:
#             shares_to_sell = int(self.shares_held * abs(action))
#             sale_val = shares_to_sell * current_price
#             fee = sale_val * self.transaction_cost_percent
#             self.balance += sale_val - fee
#             self.shares_held -= shares_to_sell

#             self.total_shares_sold += shares_to_sell
#             self.total_sales_value += sale_val

#         # update worth
#         self.net_worth = self.balance + self.shares_held * current_price
#         self.max_net_worth = max(self.max_net_worth, self.net_worth)

#         reward = self.net_worth - self.initial_balance
#         done = done or self.net_worth <= 0

#         return self._next_observation(), reward, done, False, {}

#     def render(self, mode: str = "human"):
#         profit = self.net_worth - self.initial_balance
#         print(
#             f"Step {self.current_step} | "
#             f"Bal: {self.balance:.2f} | Shares: {self.shares_held} | "
#             f"Net: {self.net_worth:.2f} | PnL: {profit:.2f}"
#         )

#     def close(self):
#         pass


# def test_model(env, model, n_tests=1000, visualize=False):
#     """
#     Test a trained model and track performance metrics, with an option to visualize the results.

#     Parameters:
#     - env: The vectorized environment (DummyVecEnv).
#     - model: The trained model (SAC).
#     - n_tests: Number of steps to test (default: 1000).
#     - visualize: Boolean flag to enable or disable visualization (default: False).

#     Returns:
#     - A dictionary containing performance metrics.
#     """
#     # Initialize metrics tracking
#     metrics = {
#         "steps": [],
#         "rewards": [],
#         "balances": [],
#         "net_worths": [],
#         "actions": [],
#     }

#     # Reset the environment before starting the tests
#     obs = env.reset()

#     # Testing loop
#     for i in range(n_tests):
#         # Record step
#         metrics["steps"].append(i)

#         # Get action from model - handle observation shape
#         obs_for_model = (
#             obs.squeeze() if len(obs.shape) > 1 and obs.shape[0] == 1 else obs
#         )

#         # Get and format action
#         action, _ = model.predict(obs_for_model, deterministic=True)

#         # Ensure action is in the correct format for the vectorized environment
#         # If action is a scalar and env expects array for 1 env
#         if np.isscalar(action) and env.action_space.shape[0] > 0:
#             action = np.array([action])

#         # If we have multiple actions but need to wrap for vectorized env
#         if len(action.shape) == 1 and env.num_envs == 1:
#             # Store the original action for metrics
#             metrics["actions"].append(action.copy())
#         else:
#             metrics["actions"].append(action)

#         # Step the environment
#         obs, reward, done, info = env.step(action)

#         # Handle reward - might be array for vectorized env
#         if isinstance(reward, (list, np.ndarray)):
#             metrics["rewards"].append(reward[0])
#         else:
#             metrics["rewards"].append(reward)

#         # Track environment metrics - safely
#         try:
#             metrics["balances"].append(env.get_attr("balance")[0])
#             metrics["net_worths"].append(env.get_attr("net_worth")[0])
#         except (AttributeError, IndexError) as e:
#             print(f"Warning: Could not get environment attributes: {e}")
#             metrics["balances"].append(
#                 metrics["balances"][-1] if metrics["balances"] else 0
#             )
#             metrics["net_worths"].append(
#                 metrics["net_worths"][-1] if metrics["net_worths"] else 0
#             )

#         if visualize:
#             env.render()

#         # Reset if episode is done
#         if isinstance(done, (list, np.ndarray)):
#             if done[0]:
#                 obs = env.reset()
#         elif done:
#             obs = env.reset()

#     # Calculate cumulative returns safely
#     if metrics["rewards"]:
#         metrics["cumulative_rewards"] = np.cumsum(metrics["rewards"])
#     else:
#         metrics["cumulative_rewards"] = []

#     # Performance summary if we have data
#     if metrics["net_worths"]:
#         initial_worth = metrics["net_worths"][0]
#         final_worth = metrics["net_worths"][-1]
#         profit = final_worth - initial_worth
#         profit_percent = (profit / initial_worth) * 100 if initial_worth else 0

#         print(f"\nPerformance Summary:")
#         print(f"Initial Net Worth: ${initial_worth:.2f}")
#         print(f"Final Net Worth: ${final_worth:.2f}")
#         print(f"Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)")

#     return metrics


# class StockTradingEnv(gym.Env):
#     metadata = {"render_modes": ["human"]}

#     def __init__(self, stock_data, transaction_cost_percent=0.005):
#         super(StockTradingEnv, self).__init__()

#         # Remove any empty DataFrames
#         self.stock_data = {
#             ticker: df for ticker, df in stock_data.items() if not df.empty
#         }
#         self.tickers = list(self.stock_data.keys())

#         if not self.tickers:
#             raise ValueError("All provided stock data is empty")

#         # Calculate the size of one stock's data
#         sample_df = next(iter(self.stock_data.values()))
#         self.n_features = len(sample_df.columns)

#         # Action space, 1 is buy, -1 is sell, 0 is hold
#         self.action_space = spaces.Box(
#             low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32
#         )

#         # Observation space at each tick of df
#         # price data for each stock + balance + shares held + net worth + max net worth + current step
#         self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
#         )

#         # State variables
#         self.initial_balance = 1000000
#         self.balance = self.initial_balance
#         self.net_worth = self.initial_balance
#         self.max_net_worth = self.initial_balance
#         self.shares_held = {ticker: 0 for ticker in self.tickers}
#         self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
#         self.total_sales_value = {ticker: 0 for ticker in self.tickers}

#         # Set the current step
#         self.current_step = 0

#         # Calculate the minimum length of data across all stocks
#         self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

#         # Transaction cost
#         self.transaction_cost_percent = transaction_cost_percent

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.balance = self.initial_balance
#         self.net_worth = self.initial_balance
#         self.max_net_worth = self.initial_balance
#         self.shares_held = {ticker: 0 for ticker in self.tickers}
#         self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
#         self.total_sales_value = {ticker: 0 for ticker in self.tickers}
#         self.current_step = 0
#         return self._next_observation(), {}

#     def _next_observation(self):
#         # initialize the frame
#         frame = np.zeros(self.obs_shape)

#         # Add stock data for each ticker
#         idx = 0
#         # Loop through each ticker
#         for ticker in self.tickers:
#             # Get the DataFrame for the current ticker
#             df = self.stock_data[ticker]
#             # If the current step is less than the length of the DataFrame, add the price data for the current step
#             if self.current_step < len(df):
#                 frame[idx : idx + self.n_features] = df.iloc[self.current_step].values
#             # Otherwise, add the last price data available
#             elif len(df) > 0:
#                 frame[idx : idx + self.n_features] = df.iloc[-1].values
#             # Move the index to the next ticker
#             idx += self.n_features

#         # Add balance, shares held, net worth, max net worth, and current step
#         frame[-4 - len(self.tickers)] = self.balance  # Balance
#         frame[-3 - len(self.tickers) : -3] = [
#             self.shares_held[ticker] for ticker in self.tickers
#         ]  # Shares held
#         frame[-3] = self.net_worth  # Net worth
#         frame[-2] = self.max_net_worth  # Max net worth
#         frame[-1] = self.current_step  # Current step

#         return frame

#     def step(self, actions):
#         if np.isscalar(actions) or (
#             isinstance(actions, np.ndarray) and actions.size == 1
#         ):
#             # If we only have one ticker, a scalar action is fine
#             if len(self.tickers) == 1:
#                 actions = np.array([actions])
#             else:
#                 # If multiple tickers but scalar action, replicate it
#                 actions = np.full(len(self.tickers), actions)

#         # update the current step
#         self.current_step += 1

#         # check if we have reached the maximum number of steps
#         if self.current_step > self.max_steps:
#             return self._next_observation(), 0, True, False, {}

#         current_prices = {}
#         # Loop through each ticker and perform the action
#         for i, ticker in enumerate(self.tickers):
#             # Get the current price of the stock
#             current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step][
#                 "close"
#             ]
#             # get the action for the current ticker
#             action = actions[i]

#             if action > 0:  # Buy
#                 # Calculate the number of shares to buy
#                 shares_to_buy = int(self.balance * action / current_prices[ticker])
#                 # Calculate the cost of the shares
#                 cost = shares_to_buy * current_prices[ticker]
#                 # Transaction cost
#                 transaction_cost = cost * self.transaction_cost_percent
#                 # Update the balance and shares held
#                 self.balance -= cost + transaction_cost
#                 # Update the total shares sold
#                 self.shares_held[ticker] += shares_to_buy

#             elif action < 0:  # Sell
#                 # Calculate the number of shares to sell
#                 shares_to_sell = int(self.shares_held[ticker] * abs(action))
#                 # Calculate the sale value
#                 sale = shares_to_sell * current_prices[ticker]
#                 # Transaction cost, fixed fees...
#                 transaction_cost = sale * self.transaction_cost_percent
#                 # Update the balance and shares held
#                 self.balance += sale - transaction_cost
#                 # Update the total shares sold
#                 self.shares_held[ticker] -= shares_to_sell
#                 # Update the shares sold
#                 self.total_shares_sold[ticker] += shares_to_sell
#                 # Update the total sales value
#                 self.total_sales_value[ticker] += sale

#         # Calculate the net worth
#         self.net_worth = self.balance + sum(
#             self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers
#         )
#         # Update the max net worth
#         self.max_net_worth = max(self.net_worth, self.max_net_worth)
#         # Calculate the reward
#         reward = self.net_worth - self.initial_balance
#         # Check if the episode is done
#         done = self.net_worth <= 0 or self.current_step >= self.max_steps

#         obs = self._next_observation()
#         return obs, reward, done, False, {}

#     def render(self, mode="human"):
#         # Print the current step, balance, shares held, net worth, and profit
#         profit = self.net_worth - self.initial_balance
#         print(f"Step: {self.current_step}")
#         print(f"Balance: {self.balance:.2f}")
#         for ticker in self.tickers:
#             print(f"{ticker} Shares held: {self.shares_held[ticker]}")
#         print(f"Net worth: {self.net_worth:.2f}")
#         print(f"Profit: {profit:.2f}")

#     def close(self):
#         pass


# def test_agent(env, agent, stock_data, n_tests=1000, visualize=False):
#     """
#     Test a single agent and track performance metrics, with an option to visualize the results.

#     Parameters:
#     - env: The trading environment.
#     - agent: The agent to be tested.
#     - stock_data: Data for the stocks in the environment.
#     - n_tests: Number of tests to run (default: 1000).
#     - visualize: Boolean flag to enable or disable visualization (default: False).

#     Returns:
#     - A dictionary containing steps, balances, net worths, and shares held.
#     """
#     # Initialize metrics tracking
#     metrics = {
#         "steps": [],
#         "datetimes": [],
#         "balances": [],
#         "net_worths": [],
#         "shares_held": {ticker: [] for ticker in stock_data.keys()},
#     }

#     # Reset the environment before starting the tests
#     obs = env.reset()

#     # If you have 3 tickers, action will be something like [0.2, -0.5, 0.0]
#     for i in range(n_tests):
#         metrics["steps"].append(i)
#         action = agent.predict(obs)
#         obs, rewards, dones, infos = env.step(action)
#         if visualize:
#             env.render()

#         # Track metrics
#         metrics["balances"].append(env.get_attr("balance")[0])
#         metrics["net_worths"].append(env.get_attr("net_worth")[0])
#         env_shares_held = env.get_attr("shares_held")[0]

#         # Update shares held for each ticker
#         for ticker in stock_data.keys():
#             if ticker in env_shares_held:
#                 metrics["shares_held"][ticker].append(env_shares_held[ticker])
#             else:
#                 metrics["shares_held"][ticker].append(
#                     0
#                 )  # Append 0 if ticker is not found

#         if dones:
#             obs = env.reset()

#     return metrics


# # function to visualize the multiple portfolio net worths ( same chart )
# def visualize_multiple_portfolio_net_worth(steps, net_worths_list, labels):
#     plt.figure(figsize=(12, 6))
#     for i, net_worths in enumerate(net_worths_list):
#         plt.plot(steps, net_worths, label=labels[i])
#     plt.title("Net Worth Over Time")
#     plt.xlabel("Steps")
#     plt.ylabel("Net Worth")
#     plt.legend()
#     plt.show()


# def test_and_visualize_agents(env, agents, training_data, n_tests=1000):
#     metrics = {}
#     for agent_name, agent in agents.items():
#         print(f"Testing {agent_name}...")
#         metrics[agent_name] = test_agent(
#             env, agent, training_data, n_tests=n_tests, visualize=True
#         )
#         print(f"Done testing {agent_name}!")

#     print("-" * 50)
#     print("All agents tested!")
#     print("-" * 50)

#     # Extract net worths for visualization
#     net_worths = [metrics[agent_name]["net_worths"] for agent_name in agents.keys()]
#     steps = next(iter(metrics.values()))[
#         "steps"
#     ]  # Assuming all agents have the same step count for simplicity

#     # Visualize the performance metrics of multiple agents
#     visualize_multiple_portfolio_net_worth(steps, net_worths, list(agents.keys()))


# SER using change point detection
# we treat changing/different market regimes as sequential tasks
# labels that have the same tag as the current regime will have higher priority in long-term memory; when there is a change point the scoring changes entirely

# considering combination of matching global distribution w/ random normal dist and matching regimes detected by center-point detection

# TODO: still need to fix/add "reward" strategy; also check coverage


# class SERReplayBuffer(BaseBuffer):
#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         strategy: str,
#         priority_queue_size: int,
#         priority_queue_percent: float,  # 10% use --> 0.1
#         device: Union[torch.device, str] = "auto",
#         n_envs: int = 1,
#         optimize_memory_usage: bool = False,
#         handle_timeout_termination: bool = True,
#     ):
#         super().__init__(
#             buffer_size, observation_space, action_space, device, n_envs=n_envs
#         )

#         # Adjust buffer size
#         self.buffer_size = max(buffer_size // n_envs, 1)

#         # Check that the replay buffer can fit into the memory
#         if psutil is not None:
#             mem_available = psutil.virtual_memory().available

#         # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
#         # see https://github.com/DLR-RM/stable-baselines3/issues/934
#         if optimize_memory_usage and handle_timeout_termination:
#             raise ValueError(
#                 "ReplayBuffer does not support optimize_memory_usage = True "
#                 "and handle_timeout_termination = True simultaneously."
#             )
#         self.optimize_memory_usage = optimize_memory_usage

#         self.observations = np.zeros(
#             (self.buffer_size, self.n_envs, *self.obs_shape),
#             dtype=observation_space.dtype,
#         )

#         if not optimize_memory_usage:
#             # When optimizing memory, `observations` contains also the next observation
#             self.next_observations = np.zeros(
#                 (self.buffer_size, self.n_envs, *self.obs_shape),
#                 dtype=observation_space.dtype,
#             )

#         self.actions = np.zeros(
#             (self.buffer_size, self.n_envs, self.action_dim),
#             dtype=self._maybe_cast_dtype(action_space.dtype),
#         )

#         self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         # Handle timeouts termination properly if needed
#         # see https://github.com/DLR-RM/stable-baselines3/issues/284
#         self.handle_timeout_termination = handle_timeout_termination
#         self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

#         if psutil is not None:
#             total_memory_usage: float = (
#                 self.observations.nbytes
#                 + self.actions.nbytes
#                 + self.rewards.nbytes
#                 + self.dones.nbytes
#             )

#             if not optimize_memory_usage:
#                 total_memory_usage += self.next_observations.nbytes

#             if total_memory_usage > mem_available:
#                 # Convert to GB
#                 total_memory_usage /= 1e9
#                 mem_available /= 1e9
#                 warnings.warn(
#                     "This system does not have apparently enough memory to store the complete "
#                     f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
#                 )

#         self.strategy = strategy
#         self.gamma = 0.99
#         self.dist_threshold = 0.5
#         self.priority_queue_size = priority_queue_size
#         self.long_term_memory = []  # (score, (obs, next_obs, action, reward, done))
#         self.priority_queue_percent = priority_queue_percent
#         self.q_net = None
#         self.q_net_target = None

#     def add(
#         self,
#         obs: np.ndarray,
#         next_obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: list[dict[str, Any]],
#     ) -> None:
#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#         if isinstance(self.observation_space, spaces.Discrete):
#             obs = obs.reshape((self.n_envs, *self.obs_shape))
#             next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

#         # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
#         action = action.reshape((self.n_envs, self.action_dim))

#         # Copy to avoid modification by reference
#         self.observations[self.pos] = np.array(obs)

#         if self.optimize_memory_usage:
#             self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
#         else:
#             self.next_observations[self.pos] = np.array(next_obs)

#         self.actions[self.pos] = np.array(action)
#         self.rewards[self.pos] = np.array(reward)
#         self.dones[self.pos] = np.array(done)

#         if self.handle_timeout_termination:
#             self.timeouts[self.pos] = np.array(
#                 [info.get("TimeLimit.truncated", False) for info in infos]
#             )

#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True
#             self.pos = 0

#         # compute score using given strategy and push to priority queue
#         # will get rid of the lowest score among the ones stored in long term mem if we exceed memory limit

#         idx = (self.pos - 1) % self.buffer_size
#         for env_idx in range(self.n_envs):
#             score = self.compute_score(
#                 obs[env_idx],
#                 next_obs[env_idx],
#                 action[env_idx],
#                 reward[env_idx],
#                 done[env_idx],
#             )
#             heapq.heappush(self.long_term_memory, (score, idx, env_idx))

#         if len(self.long_term_memory) > self.priority_queue_size:
#             heapq.heappop(self.long_term_memory)

#     def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
#         # print("sampling")
#         long_size = int(batch_size * self.priority_queue_percent)
#         fifo_size = batch_size - long_size

#         # sample from FIFO
#         if not self.optimize_memory_usage:
#             return super().sample(batch_size=batch_size, env=env)
#         # Do not sample the element with index `self.pos` as the transitions is invalid
#         # (we use only one array to store `obs` and `next_obs`)
#         if self.full:
#             fifo_inds = (
#                 np.random.randint(1, self.buffer_size, size=fifo_size) + self.pos
#             ) % self.buffer_size
#         else:
#             fifo_inds = np.random.randint(0, self.pos, size=fifo_size)

#         fifo_samples = self._get_samples(fifo_inds, env=env)

#         # sample from long term memory
#         if len(self.long_term_memory) >= long_size:
#             sampled_long_mem = random.sample(self.long_term_memory, long_size)
#         else:
#             sampled_long_mem = self.long_term_memory

#         buffer_idxs = [idx for _, idx, env in sampled_long_mem]
#         env_idxs = [env for _, idx, env in sampled_long_mem]

#         long_obs = self.observations[buffer_idxs, env_idxs]
#         if self.optimize_memory_usage:
#             long_next_obs = self.observations[
#                 (np.array(buffer_idxs) + 1) % self.buffer_size, env_idxs
#             ]
#         else:
#             long_next_obs = self.next_observations[buffer_idxs, env_idxs]

#         long_actions = self.actions[buffer_idxs, env_idxs]
#         long_rewards = self.rewards[buffer_idxs, env_idxs].reshape(-1, 1)
#         long_dones = (
#             self.dones[buffer_idxs, env_idxs]
#             * (1 - self.timeouts[buffer_idxs, env_idxs])
#         ).reshape(-1, 1)

#         long_obs = self.to_torch(self._normalize_obs(long_obs, env))
#         long_next_obs = self.to_torch(self._normalize_obs(long_next_obs, env))
#         long_actions = self.to_torch(long_actions)
#         long_rewards = self.to_torch(self._normalize_reward(long_rewards, env))
#         long_dones = self.to_torch(long_dones)

#         # combine FIFO + long-term mem
#         obs = torch.cat([fifo_samples.observations, long_obs], dim=0)
#         next_obs = torch.cat([fifo_samples.next_observations, long_next_obs], dim=0)
#         actions = torch.cat([fifo_samples.actions, long_actions], dim=0)
#         rewards = torch.cat([fifo_samples.rewards, long_rewards], dim=0)
#         dones = torch.cat([fifo_samples.dones, long_dones], dim=0)

#         return ReplayBufferSamples(obs, actions, next_obs, dones, rewards)

#     def _get_samples(
#         self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
#     ) -> ReplayBufferSamples:
#         # Sample randomly the env idx
#         env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

#         if self.optimize_memory_usage:
#             next_obs = self._normalize_obs(
#                 self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
#                 env,
#             )
#         else:
#             next_obs = self._normalize_obs(
#                 self.next_observations[batch_inds, env_indices, :], env
#             )

#         data = (
#             self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
#             self.actions[batch_inds, env_indices, :],
#             next_obs,
#             # Only use dones that are not due to timeouts
#             # deactivated by default (timeouts is initialized as an array of False)
#             (
#                 self.dones[batch_inds, env_indices]
#                 * (1 - self.timeouts[batch_inds, env_indices])
#             ).reshape(-1, 1),
#             self._normalize_reward(
#                 self.rewards[batch_inds, env_indices].reshape(-1, 1), env
#             ),
#         )
#         # print("sampling done")
#         return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

#     def compute_score(self, obs, next_obs, action, reward, done):
#         if self.strategy == "reward":
#             return float(abs(reward))  # TODO: need to implement this one
#         elif self.strategy == "distribution":
#             return float(np.random.normal())

#         elif self.strategy == "surprise":
#             with torch.no_grad():
#                 obs_tensor = self.to_torch(obs).unsqueeze(0)
#                 next_obs_tensor = self.to_torch(next_obs).unsqueeze(0)
#                 action_tensor = self.to_torch(action).long().unsqueeze(0)
#                 reward_tensor = self.to_torch(reward).unsqueeze(0)
#                 done_tensor = self.to_torch(done).unsqueeze(0).float()

#                 q_values = self.q_net(obs_tensor)
#                 q_sa = q_values.gather(1, action_tensor)

#                 next_q_values = self.q_net_target(next_obs_tensor)
#                 max_q_next = next_q_values.max(1, keepdim=True).values

#                 td_target = (
#                     reward_tensor + self.gamma * (1.0 - done_tensor) * max_q_next
#                 )
#                 td_error = torch.abs(td_target - q_sa)
#                 return float(td_error.item())

#         elif self.strategy == "coverage":
#             norm_obs = self._normalize_obs(obs, env=None).flatten()
#             # build KD tree
#             all = []
#             for _, idx, env in self.long_term_memory:
#                 exist_obs = self._normalize_obs(
#                     self.observations[idx, env], env=None
#                 ).flatten()
#                 all.append(exist_obs)

#             if len(all) == 0:
#                 return 0
#             tree = KDTree(np.stack(all))
#             neighbors = tree.query_ball_point(norm_obs, r=self.dist_threshold)
#             count = len(neighbors)
#             return -count
#         return 0

#     def set_q_nets(self, q_net, q_net_target):
#         self.q_net = q_net
#         self.q_net_target = q_net_target

#     @staticmethod
#     def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
#         """
#         Cast `np.float64` action datatype to `np.float32`,
#         keep the others dtype unchanged.
#         See GH#1572 for more information.

#         :param dtype: The original action space dtype
#         :return: ``np.float32`` if the dtype was float64,
#             the original dtype otherwise.
#         """
#         if dtype == np.float64:
#             return np.float32
#         return dtype


# selective_replay_buffer.py


# class SERReplayBuffer(ReplayBuffer):
#     """
#     Selective Experience Replay (Isele & Cosgun 2018).

#     Parameters
#     ----------
#     strategy : {'surprise','reward','distribution','coverage'}
#         Selection rule for the long-term memory.
#     long_mem_size : int
#         Capacity of episodic memory 𝔅 (<< buffer_size).
#     coverage_radius : float
#         Euclidean radius used by the coverage strategy.
#     """

#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         strategy: str = "distribution",
#         device="cpu",
#         n_envs: int = 1,
#         long_mem_size: int = 50_000,
#         coverage_radius: float = 0.05,
#         optimize_memory_usage: bool = False,
#         handle_timeout_termination: bool = True,
#         **kwargs,  # future-proof
#     ):
#         super().__init__(
#             buffer_size, observation_space, action_space, device=device, n_envs=n_envs
#         )
#         assert strategy in {"surprise", "reward", "distribution", "coverage"}
#         self.strategy, self.long_mem_size = strategy, long_mem_size
#         self.coverage_radius = coverage_radius

#         # long-term storage (lists → numpy when sampling)
#         self.L_obs, self.L_next_obs, self.L_actions = [], [], []
#         self.L_rewards, self.L_dones, self.keys = [], [], []  # keys = ranking scores
#         # extras for surprise / reward
#         self.last_gamma, self.last_qvals = None, None
#         # helper for coverage
#         self._kdtree: Optional[KDTree] = None
#         # for distribution reservoir
#         self._rand = random.Random()

#     # ─────────────────────────── public API overrides ──────────── #

#     def add(self, obs, next_obs, action, reward, done, infos=None):
#         # ➊ always store in the short FIFO (super)
#         super().add(obs, next_obs, action, reward, done, infos)

#         # --- Ensure the episodic memory receives **single-environment** transitions ---
#         # The vectorised environment returns a batch of "n_envs" transitions at each
#         # step. The built-in ReplayBuffer splits those automatically, but our custom
#         # episodic lists stored the whole batch at once, causing a dimension mismatch
#         # when sampling. We now iterate over the env dimension (if any) so that each
#         # stored transition has the same rank as ReplayBuffer samples, i.e. `(obs_dim,)`.
#         obs_arr = np.asarray(obs)
#         next_obs_arr = np.asarray(next_obs)
#         action_arr = np.asarray(action)
#         reward_arr = np.asarray(reward)
#         done_arr = np.asarray(done)

#         # Detect an extra leading dimension that corresponds to the number of parallel
#         # environments. This is true when the array rank is higher than the raw
#         # observation space rank.
#         if obs_arr.ndim > len(self.observation_space.shape):
#             # Iterate over the first dimension (n_envs)
#             for i in range(obs_arr.shape[0]):
#                 self._maybe_store(
#                     obs_arr[i],
#                     next_obs_arr[i],
#                     action_arr[i],
#                     reward_arr[i],
#                     done_arr[i],
#                 )
#         else:
#             # Single-environment case (e.g. n_envs == 1)
#             self._maybe_store(obs_arr, next_obs_arr, action_arr, reward_arr, done_arr)

#     def sample(self, batch_size: int, env=None):
#         """Return ~50 % short-term + 50 % episodic transitions."""
#         short_bs = batch_size // 2
#         long_bs = batch_size - short_bs

#         short = super().sample(short_bs, env)

#         # Debug print
#         print(f"Short obs shape: {short[0].shape}, next_obs shape: {short[1].shape}")

#         long_indices = np.random.randint(0, len(self.L_obs), size=long_bs)

#         to_torch = lambda x: torch.as_tensor(x, device=self.device)
#         long = (
#             to_torch(np.asarray(self.L_obs)[long_indices]),
#             to_torch(np.asarray(self.L_next_obs)[long_indices]),
#             to_torch(np.asarray(self.L_actions)[long_indices]),
#             to_torch(np.asarray(self.L_rewards)[long_indices]),
#             to_torch(np.asarray(self.L_dones)[long_indices]),
#         )
#         # SB3 expects a ReplayBufferSamples tuple; easiest: concatenate
#         concat = lambda a, b: torch.cat([a, b], dim=0)
#         return short.__class__(*map(concat, short, long))

#     # ─────────────────────────── internal helpers ──────────────── #

#     def _maybe_store(self, obs, next_obs, action, reward, done):
#         """Apply the chosen selection rule."""
#         if self.strategy == "distribution":
#             key = self._rand.random()
#             self._reservoir_decision(key, obs, next_obs, action, reward, done)

#         elif self.strategy == "reward":
#             self._rank_and_store(
#                 abs(float(reward)), obs, next_obs, action, reward, done
#             )

#         elif self.strategy == "surprise":
#             # quick TD-error using current networks (needs Q-values from algo)
#             if self.last_qvals is None:
#                 return  # wait until algorithm feeds td-err
#             key = abs(float(self.last_qvals))
#             self._rank_and_store(key, obs, next_obs, action, reward, done)

#         elif self.strategy == "coverage":
#             key = 0.0  # placeholder
#             if len(self.L_obs) < self.long_mem_size:
#                 self._episodic_append(key, obs, next_obs, action, reward, done)
#                 self._rebuild_kdtree()
#             else:
#                 cnt = self._kdtree.query_radius(
#                     np.asarray(obs)[None, ...], r=self.coverage_radius, count_only=True
#                 )[0]
#                 if cnt == 0:  # sparse spot → replace densest cell
#                     dense_idx = self._dense_index()
#                     self._episodic_replace(
#                         dense_idx, key, obs, next_obs, action, reward, done
#                     )
#                     self._rebuild_kdtree()

#     # ----- concrete policies for each rule -----

#     def _reservoir_decision(self, key, *transition):
#         if len(self.L_obs) < self.long_mem_size:
#             self._episodic_append(key, *transition)
#         else:  # Vitter ’85 reservoir update
#             j = self._rand.randint(0, self.num_timesteps)
#             if j < self.long_mem_size:
#                 self._episodic_replace(j, key, *transition)

#     def _rank_and_store(self, key, *transition):
#         if len(self.L_obs) < self.long_mem_size:
#             heapq.heappush(self.keys, (key, len(self.L_obs)))
#             self._episodic_append(key, *transition)
#         else:
#             if key > self.keys[0][0]:  # prefer larger score
#                 _, victim = heapq.heapreplace(self.keys, (key, self.keys[0][1]))
#                 self._episodic_replace(victim, key, *transition)

#     # ----- low-level ops -----
#     def _episodic_append(self, key, obs, next_obs, action, reward, done):
#         self.L_obs.append(obs), self.L_next_obs.append(next_obs)
#         self.L_actions.append(action)
#         self.L_rewards.append(reward), self.L_dones.append(done)
#         self.keys.append(key)

#     def _episodic_replace(self, idx, key, obs, next_obs, action, reward, done):
#         self.L_obs[idx] = obs
#         self.L_next_obs[idx] = next_obs
#         self.L_actions[idx] = action
#         self.L_rewards[idx] = reward
#         self.L_dones[idx] = done
#         self.keys[idx] = key

#     def _rebuild_kdtree(self):
#         self._kdtree = KDTree(np.asarray(self.L_obs))

#     def _dense_index(self):
#         counts = self._kdtree.query_radius(
#             np.asarray(self.L_obs), r=self.coverage_radius, count_only=True
#         )
#         return int(np.argmax(counts))


class SERReplayBuffer(BaseBuffer):
    """
    Selective Experience Replay (Isele & Cosgun 2018) built on SB3’s BaseBuffer.

    – Short-term FIFO: standard ReplayBuffer (keeps the most recent transitions)
    – Long-term memory: small reservoir selected by one of four strategies
        • ‘distribution’  – uniform reservoir-sampling
        • ‘reward’        – keep large-reward transitions
        • ‘surprise’      – keep large TD-error   (needs algo to call `set_td_error`)
        • ‘coverage’      – keep transitions that increase state-space coverage
    When sampling we mix 50 % short-term + 50 % episodic.

    Parameters
    ----------
    buffer_size        : capacity of the short-term FIFO
    long_mem_size      : capacity of the episodic memory (≪ buffer_size)
    strategy           : selection rule
    coverage_radius    : Euclidean radius used by the ‘coverage’ rule
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="cpu",
        n_envs: int = 1,
        *,
        strategy: str = "distribution",
        long_mem_size: int = 50_000,
        coverage_radius: float = 0.05,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            # optimize_memory_usage=optimize_memory_usage,
            # handle_timeout_termination=handle_timeout_termination,
        )

        assert strategy in {"surprise", "reward", "distribution", "coverage"}
        self.strategy = strategy
        self.long_mem_size = long_mem_size
        self.coverage_radius = coverage_radius
        self.n_steps = 0

        # Short-term buffer (uses SB3 implementation)
        self.short = ReplayBuffer(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # Episodic storage
        self.L_obs, self.L_next_obs, self.L_actions = [], [], []
        self.L_rewards, self.L_dones, self.keys = [], [], []  # `keys` = ranking scores
        self._kdtree: Optional[KDTree] = None  # for ‘coverage’
        self._rand = random.Random()  # for ‘distribution’

        # For ‘surprise’
        self.last_td_error: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseBuffer API
    # ------------------------------------------------------------------
    def add(self, obs, next_obs, action, reward, done, infos=None):
        """Store transition in FIFO and maybe in episodic memory"""

        # 1) always store in short-term FIFO
        self.short.add(obs, next_obs, action, reward, done, infos)

        # 2) update bookkeeping expected by BaseBuffer
        self.pos = (self.pos + 1) % self.buffer_size
        self.full |= self.pos == 0
        self.n_steps += 1

        # maybe store in longterm memory
        obs = np.asarray(obs)
        batched = obs.ndim > len(self.observation_space.shape)
        for i in range(obs.shape[0] if batched else 1):
            idx = i if batched else slice(None)
            self._maybe_store(
                obs[idx],
                np.asarray(next_obs)[idx],
                np.asarray(action)[idx],
                np.asarray(reward)[idx],
                np.asarray(done)[idx],
            )

    # def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
    #     """50 % recent transitions + 50 % episodic transitions"""
    #     short_bs = batch_size // 2
    #     long_bs = batch_size - short_bs

    #     short = self.short.sample(short_bs, env)
    #     long = self._sample_long(long_bs)

    #     cat = lambda a, b: torch.cat([a, b], dim=0)
    #     return ReplayBufferSamples(
    #         observations=cat(short.observations, long.observations),
    #         actions=cat(short.actions, long.actions),
    #         next_observations=cat(short.next_observations, long.next_observations),
    #         dones=cat(short.dones, long.dones),
    #         rewards=cat(short.rewards, long.rewards),
    #     )

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """50% recent transitions + 50% episodic transitions"""
        short_bs = batch_size // 2
        long_bs = batch_size - short_bs

        # Get samples from short-term memory
        short = self.short.sample(short_bs, env)

        # Get samples from long-term memory using _get_samples
        if long_bs > 0 and len(self.L_obs) > 0:
            # Sample random indices from episodic memory
            long_inds = np.random.randint(0, len(self.L_obs), size=long_bs)
            long = self._get_samples(long_inds, env)
        else:
            # Fall back to short-term only if no episodic memory
            return short

        # Concatenate samples
        cat = lambda a, b: torch.cat([a, b], dim=0)
        return ReplayBufferSamples(
            observations=cat(short.observations, long.observations),
            actions=cat(short.actions, long.actions),
            next_observations=cat(short.next_observations, long.next_observations),
            dones=cat(short.dones, long.dones),
            rewards=cat(short.rewards, long.rewards),
        )

    # ------------------------------------------------------------------
    # Episodic-memory helpers
    # ------------------------------------------------------------------
    # def _sample_long(self, n: int) -> ReplayBufferSamples:
    #     idx = np.random.randint(0, len(self.L_obs), size=n)
    #     t = lambda x: torch.as_tensor(np.asarray(x)[idx], device=self.device)
    #     return ReplayBufferSamples(
    #         observations=t(self.L_obs),
    #         actions=t(self.L_actions),
    #         next_observations=t(self.L_next_obs),
    #         dones=t(self.L_dones),
    #         rewards=t(self.L_rewards),
    #     )
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """Get samples from our episodic memory with proper normalization"""
        if len(batch_inds) == 0 or len(self.L_obs) == 0:
            # Empty sample case - return an empty sample with the right structure
            empty_data = (
                np.zeros((0,) + self.observation_space.shape, dtype=np.float32),
                np.zeros((0, self.action_dim), dtype=np.float32),
                np.zeros((0,) + self.observation_space.shape, dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, empty_data)))

        # Stack samples into arrays
        obs_array = np.stack([self.L_obs[i] for i in batch_inds])
        actions_array = np.stack([self.L_actions[i] for i in batch_inds])
        next_obs_array = np.stack([self.L_next_obs[i] for i in batch_inds])
        dones_array = np.stack([self.L_dones[i] for i in batch_inds]).reshape(-1, 1)
        rewards_array = np.stack([self.L_rewards[i] for i in batch_inds]).reshape(-1, 1)

        # Create normalized data tuple - similar to ReplayBuffer._get_samples
        data = (
            self._normalize_obs(obs_array, env),
            actions_array,
            self._normalize_obs(next_obs_array, env),
            dones_array,
            self._normalize_reward(rewards_array, env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _maybe_store(self, obs, next_obs, action, reward, done):
        if self.strategy == "distribution":
            self._reservoir_decision(
                self._rand.random(), obs, next_obs, action, reward, done
            )

        elif self.strategy == "reward":
            self._rank_and_store(
                abs(float(reward)), obs, next_obs, action, reward, done
            )

        elif self.strategy == "surprise":
            if self.last_td_error is None:
                return
            self._rank_and_store(
                abs(float(self.last_td_error)), obs, next_obs, action, reward, done
            )

        elif self.strategy == "coverage":
            key = 0.0  # unused but kept for uniform API
            if len(self.L_obs) < self.long_mem_size:
                self._episodic_append(key, obs, next_obs, action, reward, done)
                self._rebuild_kdtree()
            else:
                cnt = self._kdtree.query_radius(
                    np.asarray(obs)[None, ...], r=self.coverage_radius, count_only=True
                )[0]
                if cnt == 0:  # sparse region → replace densest cell
                    dense_idx = self._dense_index()
                    self._episodic_replace(
                        dense_idx, key, obs, next_obs, action, reward, done
                    )
                    self._rebuild_kdtree()

    # ---- selection-rule primitives ----------------------------------
    def _reservoir_decision(self, key, *tr):
        if len(self.L_obs) < self.long_mem_size:
            self._episodic_append(key, *tr)
        else:
            j = self._rand.randint(0, self.n_steps)
            if j < self.long_mem_size:
                self._episodic_replace(j, key, *tr)

    def _rank_and_store(self, key, *tr):
        if len(self.L_obs) < self.long_mem_size:
            heapq.heappush(self.keys, (key, len(self.L_obs)))
            self._episodic_append(key, *tr)
        else:
            if key > self.keys[0][0]:  # prefer larger score
                _, idx = heapq.heapreplace(self.keys, (key, self.keys[0][1]))
                self._episodic_replace(idx, key, *tr)

    # ---- low-level memory ops ---------------------------------------
    def _episodic_append(self, key, obs, next_obs, action, reward, done):
        self.L_obs.append(obs)
        self.L_next_obs.append(next_obs)
        self.L_actions.append(action)
        self.L_rewards.append(reward)
        self.L_dones.append(done)
        self.keys.append(key)

    def _episodic_replace(self, idx, key, obs, next_obs, action, reward, done):
        self.L_obs[idx] = obs
        self.L_next_obs[idx] = next_obs
        self.L_actions[idx] = action
        self.L_rewards[idx] = reward
        self.L_dones[idx] = done
        self.keys[idx] = key

    # ---- helpers for ‘coverage’ -------------------------------------
    def _rebuild_kdtree(self):
        self._kdtree = KDTree(np.asarray(self.L_obs))

    def _dense_index(self):
        counts = self._kdtree.query_radius(
            np.asarray(self.L_obs), r=self.coverage_radius, count_only=True
        )
        return int(np.argmax(counts))

    # ------------------------------------------------------------------
    # Optional hook for the algorithm to pass TD-errors (‘surprise’)
    # ------------------------------------------------------------------
    def set_td_error(self, td_error: float):
        self.last_td_error = td_error


def train(env_spawner, eval_env_spawner, config):
    print("Initializing...")

    # Unpack config
    log_dir = config["log_dir"]
    n_envs = config["n_envs"]
    ckpt = config.get("checkpoint", None)
    policy_args = config["policy_args"]
    algo_kwargs = config["algo_kwargs"]
    checkpoint_freq = config["checkpoint_freq"]
    eval_freq = config["eval_freq"]
    n_eval_episodes = config["n_eval_episodes"]
    n_train_timesteps = config["n_train_timesteps"]
    verbose_training = config["verbose_training"]

    # Setup
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = VecMonitor(
        DummyVecEnv(
            [env_spawner] * n_envs,
        )
    )
    eval_env = VecMonitor(DummyVecEnv([eval_env_spawner]))

    # if ckpt is None:
    #     model = DQN(
    #         policy="MlpPolicy",
    #         env=env,
    #         policy_kwargs=policy_args,
    #         **algo_kwargs,
    #     )
    # else:
    #     model = DQN.load(
    #         path=ckpt,
    #         env=env,
    #     )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Algorithm kwargs:")
    for key, value in algo_kwargs.items():
        print(f"  {key}: {value}")

    if ckpt is None:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_args,
            **algo_kwargs,
            device=device,
        )
    else:
        model = SAC.load(
            path=ckpt,
            env=env,
        )

    if isinstance(model.replay_buffer, SERReplayBuffer):
        if isinstance(model, SAC):
            model.replay_buffer.set_q_nets(model.critic.qf0, model.critic_target.qf0)
        else:
            model.replay_buffer.set_q_nets(model.q_net, model.q_net_target)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix=os.path.basename(log_dir),
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # Train
    print("Training...")
    # model.learn(
    #     total_timesteps=n_train_timesteps,
    #     callback=[
    #         eval_callback,
    #         checkpoint_callback,
    #     ],
    #     log_interval=1 if verbose_training else None,
    #     reset_num_timesteps=False,
    #     progress_bar=True if verbose_training else False,
    # )
    try:
        model.learn(
            total_timesteps=n_train_timesteps,
            callback=[
                eval_callback,
                checkpoint_callback,
            ],
            log_interval=1 if verbose_training else None,
            reset_num_timesteps=False,
            progress_bar=True if verbose_training else False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user! Saving current model...")
        interrupted_save_path = os.path.join(
            log_dir, f"interrupted_model_{int(time.time())}"
        )
        model.save(interrupted_save_path)
        print(f"Model saved to {interrupted_save_path}")
    finally:
        # Optional: Close environments or clean up if needed
        env.close()
        eval_env.close()
        print("Training finished or interrupted.")

    return model  # Return the model


def plot_evaluations(log_dir):
    evaluations = np.load(os.path.join(log_dir, "evaluations.npz"))
    timesteps = evaluations["timesteps"]
    results_all = evaluations["results"]  # shape: (eval_rounds, episodes_per_eval)

    mean_returns = np.mean(results_all, axis=1)
    std_returns = np.std(results_all, axis=1)
    num_episodes_averaged = results_all.shape[1]
    print(
        "Evaluations: min: {}, max: {}, std: {}".format(
            np.min(results_all), np.max(results_all), np.std(results_all)
        )
    )

    plt.figure(figsize=(5, 5))
    plt.plot(timesteps, mean_returns, label="Mean Return")
    plt.fill_between(
        timesteps,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.3,
        label="Std Dev",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Returns (averaged over {} episodes)".format(num_episodes_averaged))
    plt.title("Evaluation Returns")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(log_dir, "evaluations.png"))
    plt.show()


def add_trend_id(df, n_regimes=32, window_size=13 * 5 * 6.5, n_clusters=5):
    # GPU availability
    try:
        gpu_available = cuda.is_available() and bool(cuda.list_devices())
    except Exception:
        gpu_available = False

    # Resample to 1Hour
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_resampled = df.resample("1h", on="datetime")["close"].last().to_frame()
    df_resampled["log_return"] = np.log(df_resampled["close"]).diff().fillna(0)
    signal = df_resampled["log_return"].values.astype(np.float64)

    # Quarter‑year window: roughly 13 weeks of trading (≈6.5 h/day)
    window_size = int(13 * 5 * 6.5)  # ≈ 422 hourly bars
    n_regimes = 32

    # Matrix Profile
    mp_start_time = time.time()
    if gpu_available:
        try:
            matrix_profile = stumpy.gpu_stump(signal, m=window_size)
        except Exception:
            matrix_profile = stumpy.stump(signal, m=window_size)
    else:
        matrix_profile = stumpy.stump(signal, m=window_size)
    print(f"[MP] done in {time.time() - mp_start_time:.2f}s")

    # FLUSS segmentation
    fluss_start_time = time.time()
    nn_idx = matrix_profile[:, 1]
    try:
        _, regimes = stumpy.fluss(
            nn_idx, L=window_size, n_regimes=n_regimes, excl_factor=1
        )
        change_points = sorted(set(regimes.tolist() + [len(signal)]))
    except Exception:
        change_points = []
    print(
        f"[FLUSS] {max(0, len(change_points)-1)} segments in {time.time() - fluss_start_time:.2f}s"
    )

    # Feature extraction
    feat_start_time = time.time()
    feats, segs = [], []
    prev = 0
    for end in tqdm(change_points):
        seg = df.iloc[prev:end]
        if not seg.empty:
            feats.append(
                [
                    seg["log_return"].mean(),
                    seg["log_return"].std(),
                    seg["volatility_atr"].mean(),
                    seg["volatility_bbw"].mean(),
                    seg["volatility_dcw"].mean(),
                    len(seg),
                ]
            )
            segs.append((prev, end))
        prev = end
    print(f"[FEAT] done in {time.time() - feat_start_time:.2f}s")
    # Clustering & assignment
    if feats:
        # 1) KMeans on your segment‐level features
        t_km = time.time()
        labels = KMeans(n_clusters=n_clusters, random_state=12, n_init=10).fit_predict(
            np.array(feats)
        )
        print(f"[KMEANS] {n_clusters} clusters in {time.time() - t_km:.2f}s")
        print(labels)

        # 2) Exclude the sentinel CP == len(signal)
        safe_cps = [cp for cp in change_points if cp < len(signal)]
        cp_idxs = [min(cp, len(df_resampled) - 1) for cp in safe_cps]
        break_times = df_resampled.index[cp_idxs]

        # 3) Build your datetime‐bins so you get exactly len(labels) intervals
        start = df["datetime"].min()
        end = df["datetime"].max() + pd.Timedelta(seconds=1)
        bins = [start] + list(break_times) + [end]

        # 4) Map each original row into one of those len(labels) bins
        bin_idx = pd.cut(
            df["datetime"], bins=bins, right=False, labels=False
        ).to_numpy()

        # 5) Assign the true KMeans cluster to each row
        df["trend_id"] = labels[bin_idx]

        df_plot = df.sort_values("datetime")

    # PLOTTING
    # safe break‐point timestamps (no sentinel)
    safe_cps = [cp for cp in change_points if cp < len(signal)]
    cp_idxs = [min(cp, len(df_resampled) - 1) for cp in safe_cps]
    break_times = list(df_resampled.index[cp_idxs])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_plot["datetime"], df_plot["close"], color="lightgray", lw=1)

    # draw only on real label changes
    for i, bt in enumerate(break_times[:-1]):
        if labels[i] != labels[i + 1]:
            ax.axvline(bt, color="black", linestyle="--", alpha=0.5)

    # compute mid‐y
    ymin, ymax = df_plot["close"].min(), df_plot["close"].max()
    y_mid = ymin + (ymax - ymin) * 0.5

    # build segment boundaries in datetime
    starts = [df_plot["datetime"].min()] + break_times
    ends = break_times + [df_plot["datetime"].max()]

    # annotate each segment label at its time‐midpoint
    for seg_idx, lbl in enumerate(labels):
        t0 = starts[seg_idx]
        t1 = ends[seg_idx]
        mid_t = t0 + (t1 - t0) / 2
        ax.text(
            mid_t, y_mid, str(lbl), fontsize=26, color="red", ha="center", va="center"
        )

    ax.set_ylim(ymin, ymax)
    ax.set_title("Close Price with Regime IDs")
    plt.tight_layout()
    plt.show()
    print(break_times)

    return break_times
