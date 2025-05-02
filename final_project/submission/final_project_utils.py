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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor


import heapq, random, numpy as np
from typing import Optional, Sequence
from stable_baselines3.common.buffers import ReplayBuffer
from sklearn.neighbors import KDTree

from finrl.agents.stablebaselines3.models import DRLAgent

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


class SERReplayBuffer(BaseBuffer):
    """
    Selective Experience Replay (Isele & Cosgun 2018) built on BaseBuffer.

    – Short-term FIFO: standard ReplayBuffer (keeps the most recent transitions)
    – Long-term memory: small reservoir selected by one of four strategies
        • ‘distribution’  – uniform reservoir-sampling
        • ‘reward’        – keep large-reward transitions
        • ‘surprise’      – keep large TD-error
        • ‘coverage’      – keep transitions that increase state-space coverage
    When sampling we mix 50 % short-term + 50 % episodic.

    Parameters
    ----------
    buffer_size        : capacity of the short-term FIFO
    long_mem_size      : capacity of the episodic memory (≪ buffer_size)
    strategy           : selection rule
    coverage_radius    : Euclidean radius used by the coverage rule
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="cpu",  # cpu sometimes faster than gpu
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

        # fifo short term memory
        self.short = ReplayBuffer(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # Episodic/longterm storage
        self.L_obs, self.L_next_obs, self.L_actions = [], [], []
        self.L_rewards, self.L_dones, self.keys = [], [], []
        self._kdtree: Optional[KDTree] = None
        self._rand = random.Random()

        # For ‘surprise’
        self.last_td_error: Optional[float] = None
        self.q_net = None
        self.q_net_target = None

    # BaseBuffer API
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

    # half fifo half longterm
    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        short_bs = batch_size // 2
        long_bs = batch_size - short_bs
        # fifo samples
        short = self.short.sample(short_bs, env)
        # longterm samples
        if long_bs > 0 and len(self.L_obs) > 0:
            long_inds = np.random.randint(0, len(self.L_obs), size=long_bs)
            long = self._get_samples(long_inds, env)
        else:
            return short

        cat = lambda a, b: torch.cat([a, b], dim=0)
        return ReplayBufferSamples(
            observations=cat(short.observations, long.observations),
            actions=cat(short.actions, long.actions),
            next_observations=cat(short.next_observations, long.next_observations),
            dones=cat(short.dones, long.dones),
            rewards=cat(short.rewards, long.rewards),
        )

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """Get samples from our episodic memory with proper normalization"""
        if len(batch_inds) == 0 or len(self.L_obs) == 0:
            empty_data = (
                np.zeros((0,) + self.observation_space.shape, dtype=np.float32),
                np.zeros((0, self.action_dim), dtype=np.float32),
                np.zeros((0,) + self.observation_space.shape, dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, empty_data)))

        # stack samples into arrays
        obs_array = np.stack([self.L_obs[i] for i in batch_inds])
        actions_array = np.stack([self.L_actions[i] for i in batch_inds])
        next_obs_array = np.stack([self.L_next_obs[i] for i in batch_inds])
        dones_array = np.stack([self.L_dones[i] for i in batch_inds]).reshape(-1, 1)
        rewards_array = np.stack([self.L_rewards[i] for i in batch_inds]).reshape(-1, 1)

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
            td_error = self._calculate_td_error(obs, next_obs, action, reward, done)
            if td_error is None:
                # Fall back to externally provided TD error
                if self.last_td_error is None:
                    return
                td_error = self.last_td_error
            self._rank_and_store(
                abs(float(td_error)), obs, next_obs, action, reward, done
            )

        elif self.strategy == "coverage":
            key = 0.0
            if len(self.L_obs) < self.long_mem_size:
                self._episodic_append(key, obs, next_obs, action, reward, done)
                self._rebuild_kdtree()
            else:
                cnt = self._kdtree.query_radius(
                    np.asarray(obs)[None, ...], r=self.coverage_radius, count_only=True
                )[0]
                if cnt == 0:
                    dense_idx = self._dense_index()
                    self._episodic_replace(
                        dense_idx, key, obs, next_obs, action, reward, done
                    )
                    self._rebuild_kdtree()

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
            if key > self.keys[0][0]:
                _, idx = heapq.heapreplace(self.keys, (key, self.keys[0][1]))
                self._episodic_replace(idx, key, *tr)

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

    # coverage helpers
    def _rebuild_kdtree(self):
        self._kdtree = KDTree(np.asarray(self.L_obs))

    def _dense_index(self):
        counts = self._kdtree.query_radius(
            np.asarray(self.L_obs), r=self.coverage_radius, count_only=True
        )
        return int(np.argmax(counts))

    # surprise helpers
    def set_q_nets(self, q_net, q_net_target):
        """Store Q-networks to compute TD errors for 'surprise' strategy"""
        self.q_net = q_net
        self.q_net_target = q_net_target

    def _q_to_scalar(
        self, q_out: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if isinstance(q_out, (tuple, list)):
            q_out = torch.cat(q_out, dim=1)  # shape [batch, n_critics]
            q_out, _ = torch.min(q_out, dim=1, keepdim=True)
        return q_out  # shape [batch, 1]

    def _calculate_td_error(
        self, obs, next_obs, action, reward, done, gamma: float = 0.99
    ):
        if self.q_net is None or self.q_net_target is None:
            return None

        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            next_obs_t = torch.as_tensor(
                next_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action_t = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            reward_t = torch.as_tensor(
                [reward], dtype=torch.float32, device=self.device
            )
            done_t = torch.as_tensor([done], dtype=torch.float32, device=self.device)

            current_q = self._q_to_scalar(self.q_net(obs_t, action_t))
            target_q = self._q_to_scalar(self.q_net_target(next_obs_t, action_t))
            target_q = reward_t + (1 - done_t) * gamma * target_q

            return float(torch.abs(current_q - target_q).cpu().item())


def train_rl(agent, env_train, env_eval, algo, config):
    print("Initializing...")

    log_dir = config["log_dir"]
    ckpt = config.get("checkpoint", None)
    model_kwargs = config["model_kwargs"]
    checkpoint_freq = config["checkpoint_freq"]
    eval_freq = config["eval_freq"]
    n_eval_episodes = config["n_eval_episodes"]
    n_train_timesteps = config["n_train_timesteps"]
    verbose_training = config["verbose_training"]
    policy = config.get("policy", "MlpPolicy")
    policy_kwargs = config.get("policy_kwargs", None)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model = agent.get_model(
        algo,
        policy=policy,
        policy_kwargs=policy_kwargs,
        model_kwargs=model_kwargs,
        tensorboard_log=log_dir,
    )

    # if ckpt is not None:
    #     model = model.load(ckpt, env=env_train)

    # SER support
    if hasattr(model, "replay_buffer") and hasattr(model.replay_buffer, "set_q_nets"):
        q_net = getattr(model, "q_net", None) or getattr(model, "critic", None)
        q_net_target = getattr(model, "q_net_target", None) or getattr(
            model, "critic_target", None
        )

        if q_net is not None and q_net_target is not None:
            model.replay_buffer.set_q_nets(q_net, q_net_target)

    # Logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix=os.path.basename(log_dir),
        verbose=0,
    )

    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )

    print("Training...")
    try:
        model.learn(
            total_timesteps=n_train_timesteps,
            callback=[checkpoint_callback, eval_callback, ProgressBarCallback()],
            log_interval=1 if verbose_training else None,
            reset_num_timesteps=False,
            # progress_bar=True if verbose_training else False,
        )
    except BaseException as error:
        print(f"Error during training: {error}")
        raise

    return model


def plot_model(model, environment, deterministic=False, debug_print=False):
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=environment, deterministic=deterministic
    )

    def plot_df_account_value(df):
        print("Backtesting complete.")

        print("Processing results...")
        df = df.set_index(df.columns[0])
        df = pd.DataFrame({"SAC": df["account_value"]})

        print("Plotting results...")
        plt.rcParams["figure.figsize"] = (15, 5)
        plt.figure()
        df.plot()
        plt.title(f"Backtesting Results")
        plt.xlabel("Date/Time")
        plt.ylabel("Account Value")
        plt.grid(True)
        plt.show()

    # each cell in action col is a list
    def plot_df_actions(df_actions):
        print("Backtesting complete.")
        print("Processing actions...")

        # Expand list cells into individual columns (one per asset)
        actions_expanded = pd.DataFrame(
            df_actions["actions"].to_list(), index=df_actions["date"]
        )

        # Set proper column names (Asset 1, Asset 2, ...)
        actions_expanded.columns = [
            f"Asset_{i+1}" for i in range(actions_expanded.shape[1])
        ]

        # print sum of each column
        if debug_print:
            print("Sum of actions:")
            print(actions_expanded.sum())
            print(actions_expanded.head(10000))

        # Plotting results
        print("Plotting results...")
        plt.figure(figsize=(15, 5))
        actions_expanded.plot(ax=plt.gca(), marker="o", alpha=0.7)
        plt.title("Agent Actions (Buy/Sell per Asset)")
        plt.xlabel("Date/Time")
        plt.ylabel("Shares Δ (buy > 0, sell < 0)")
        plt.grid(True)
        plt.show()

    plot_df_account_value(df_account_value)
    plot_df_actions(df_actions)


def add_trend_id(
    df, n_regimes=32, window_size=13 * 5 * 6.5, n_clusters=5, resample="1h"
):
    # GPU availability
    try:
        gpu_available = cuda.is_available() and bool(cuda.list_devices())
    except Exception:
        gpu_available = False

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"]).diff().fillna(0)

    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Resample to 1Hour
    if resample:
        df_resampled = df.resample(resample, on="datetime")["close"].last().to_frame()
        df_resampled["log_return"] = np.log(df_resampled["close"]).diff().fillna(0)
        signal = df_resampled["log_return"].values.astype(np.float64)
    else:
        signal = df["log_return"].values.astype(np.float64)

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
