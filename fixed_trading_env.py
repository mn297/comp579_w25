import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv  # Or ForexEnv, TradingEnv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import os
from datetime import datetime

# --- Import yfinance ---
import yfinance as yf

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer  # Default FIFO-like buffer
# Placeholder for your future custom buffer
# from replay_buffers.selective_replay import SelectiveReplayBuffer

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress specific yfinance warnings if needed
# logging.getLogger('yfinance').setLevel(logging.ERROR)


# --- Configuration ---
config = {
    "env_id": "stocks-v0",  # ID for StocksEnv in gym_anytrading

    # --- Data Source Configuration ---
    "ticker": "MSFT",       # Stock ticker (e.g., Microsoft)
    "start_date": "2018-01-01",  # Data start date
    "end_date": "2023-12-31",   # Data end date
    # --- End Data Source Config ---

    "window_size": 15,      # Observation window size (increased slightly)
    "start_tick": None,     # Starting index in data (None = beginning, respecting window_size)
    "end_tick": None,       # Ending index in data (None = end)
    "initial_balance": 10000,  # Starting cash

    "use_ser": False,        # Set to True to use SER buffer later
    "ser_strategy": "reward",  # 'surprise', 'reward', 'global_match', 'coverage'

    "sac_policy": "MlpPolicy",
    "total_timesteps": 100000,  # Increased timesteps for more realistic data
    "learning_rate": 3e-4,   # Default SAC learning rate
    "buffer_size": 50000,    # Increased Replay buffer size
    "batch_size": 256,       # SAC batch size
    "tau": 0.005,            # SAC target smoothing coefficient
    "gamma": 0.99,           # Discount factor

    "log_interval": 1000,    # Log training progress interval
    "model_save_path": "models/",  # Directory to save trained models
    "results_path": "results/",   # Directory for results/plots
}

# --- Placeholder for Change Point Detection ---
def detect_change_points(df, method='pelt'):
    logging.info("Change Point Detection placeholder called.")
    return []

# --- Placeholder for Custom Selective Experience Replay Buffer ---
class SelectiveReplayBuffer(ReplayBuffer):
    def __init__(self, *args, prioritization_strategy="reward", **kwargs):
        super().__init__(*args, **kwargs)
        self.prioritization_strategy = prioritization_strategy
        self.task_id = 0
        logging.info(f"Initialized SelectiveReplayBuffer (Placeholder) with strategy: {self.prioritization_strategy}")
    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=infos)
    def sample(self, batch_size, env=None):
        logging.debug(f"SER Sampling (Placeholder) using strategy: {self.prioritization_strategy}")
        return super().sample(batch_size=batch_size, env=env)
    def set_task_id(self, task_id):
        self.task_id = task_id
        logging.info(f"SER Buffer task ID set to: {self.task_id}")


# --- Helper Function to Create Environment ---
def make_custom_env(df, window_size, start_tick, end_tick, initial_balance, task_id=0):
    if start_tick is None:
        start_tick = window_size
    if end_tick is None:
        end_tick = len(df) - 1

    if start_tick >= end_tick:
        raise ValueError(f"start_tick ({start_tick}) must be less than end_tick ({end_tick})")
    if start_tick < window_size:
        logging.warning(f"start_tick ({start_tick}) is less than window_size ({window_size}). Adjusting start_tick.")
        start_tick = window_size

    logging.info(f"Creating env for ticks: {start_tick} to {end_tick} (Total data points: {len(df)})")

    env = gym.make(
        config["env_id"],
        df=df,
        window_size=window_size,
        frame_bound=(start_tick, end_tick),
        # render_mode='human' # Use 'human' to watch, 'rgb_array' or None for faster training
    )
    return env

# --- Main Training Script ---
if __name__ == "__main__":
    # 1. Load Data using yfinance
    logging.info(f"Downloading {config['ticker']} data from {config['start_date']} to {config['end_date']}...")
    try:
        df = yf.download(config['ticker'],
                        start=config['start_date'],
                        end=config['end_date'],
                        progress=True)  # Show download progress

        if df.empty:
            raise ValueError(f"No data found for ticker {config['ticker']} in the specified date range.")

        # Ensure required columns are present (yfinance usually names them correctly)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Downloaded data missing required columns. Found: {df.columns.tolist()}")

        # Handle potential missing values (e.g., non-trading days might yield NaNs if requesting specific dates)
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError("DataFrame became empty after dropping NaNs.")

        # Fix for the dimension mismatch issue - Ensure Close column is properly extracted
        df_processed = df.copy()
        
        # Debugging info to identify issues
        logging.info(f"Data downloaded successfully. Shape: {df_processed.shape}")
        logging.info(f"Data Head:\n{df_processed.head()}")
        logging.info(f"Data columns: {df_processed.columns.tolist()}")
        
        # Check if Close is actually a Series or a single element
        if 'Close' in df_processed.columns:
            close_values = df_processed['Close'].values
            logging.info(f"Close column shape: {close_values.shape}")
        else:
            logging.error("No 'Close' column found in the DataFrame")
            exit()

    except Exception as e:
        logging.error(f"Failed to download or process data: {e}", exc_info=True)
        # Exit if data loading fails
        exit()

    # 2. (Optional) Perform Change Point Detection (on the full dataset)
    change_points = detect_change_points(df_processed)
    # For now, we'll train on the whole dataset as one task

    # 3. Create Environment
    # Using the full dataset loaded via yfinance
    train_env = make_custom_env(
        df=df_processed,
        window_size=config["window_size"],
        start_tick=config["start_tick"],  # Will default based on window_size
        end_tick=config["end_tick"],      # Will default to len(df) - 1
        initial_balance=config["initial_balance"]
    )
    # Wrap in a VecEnv for SB3
    train_vec_env = make_vec_env(lambda: train_env, n_envs=1)

    # 4. Setup Replay Buffer
    if config["use_ser"]:
        buffer_class = SelectiveReplayBuffer
        buffer_kwargs = dict(prioritization_strategy=config["ser_strategy"])
    else:
        buffer_class = ReplayBuffer
        buffer_kwargs = {}

    # 5. Initialize Agent
    model = SAC(
        config["sac_policy"],
        train_vec_env,
        verbose=1,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        replay_buffer_class=buffer_class,
        replay_buffer_kwargs=buffer_kwargs,
        tensorboard_log="./tensorboard_logs/",  # For visualizing training
        # seed=42 # Set for reproducibility if needed
    )

    # 6. Train Agent
    logging.info(f"Starting training for {config['total_timesteps']} timesteps...")
    start_time = datetime.now()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            log_interval=config["log_interval"],  # Log every N calls to `learn()`
            tb_log_name=f"SAC_{'SER' if config['use_ser'] else 'FIFO'}_{config['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
    finally:
        # Ensure environment is closed even if error occurs
        try:
            train_vec_env.close()
        except Exception as env_close_e:
            logging.error(f"Error closing environment: {env_close_e}")

    end_time = datetime.now()
    logging.info(f"Training finished. Duration: {end_time - start_time}")

    # 7. Save Model
    os.makedirs(config["model_save_path"], exist_ok=True)
    model_name = f"sac_{'ser' if config['use_ser'] else 'fifo'}_{config['ticker']}_{config['total_timesteps']}steps_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_path = os.path.join(config["model_save_path"], model_name)
    model.save(save_path)
    logging.info(f"Model saved to {save_path}.zip")  # SB3 automatically adds .zip

    logging.info("Script execution finished with real data.")
