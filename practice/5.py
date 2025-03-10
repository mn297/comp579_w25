import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Create directories
log_dir = "./logs/"
tb_log_dir = "./tb_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tb_log_dir, exist_ok=True)

# Create environment
env = make_vec_env("CartPole-v1", n_envs=4)
eval_env = make_vec_env("CartPole-v1", n_envs=1)

# Create callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=500,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path=log_dir,
    name_prefix="ppo_cartpole"
)

callback = CallbackList([checkpoint_callback, eval_callback])

# Create and train model with TensorBoard logging
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=tb_log_dir
)

model.learn(
    total_timesteps=10000,
    callback=callback,
    tb_log_name="ppo_run"
)

print("Training complete! Now you can run TensorBoard to see visualizations:")
print("tensorboard --logdir ./tb_logs/")