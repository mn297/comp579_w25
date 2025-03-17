import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
from tqdm import tqdm


# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# NEURAL NETWORK POLICY AND VALUE APPROXIMATION
class ZNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ZNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # z(s) = MLP(s, θ) as mentioned in the assignment
        z_s = self.fc3(x)
        return z_s


# approximates the state-value function using NN
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # V(s, w) as mentioned in the assignment
        value = self.fc3(x)
        return value


# BOLTZMANN POLICY
class BoltzmannPolicy:
    def __init__(self, policy_network, temperature=1.0):
        self.policy_network = policy_network
        self.temperature = temperature

    def select_action(self, state):
        # forward ZNetwork
        z_s = self.policy_network(state)

        # Apply Boltzmann distribution (formula from the assignment)
        # π(a|s) = exp(z(s,a)/T) / Σ_a∈A exp(z(s,a)/T)
        logits = z_s / self.temperature
        probs = F.softmax(logits, dim=-1)

        # categorical distribution and sample action, exmaple Categorical(probs: tensor([0.1, 0.2, 0.7]))
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action), probs.detach().numpy()


# REINFORCE, this is the agent that will train the NN
class REINFORCE:
    def __init__(
        self,
        state_dim,
        action_dim,
        temperature=1.0,
        lr=0.001,
        gamma=0.99,
        decreasing_temp=False,
        temp_decay=0.999,
    ):
        self.policy_network = ZNetwork(state_dim, action_dim)
        self.policy = BoltzmannPolicy(self.policy_network, temperature)
        # Use Adam instead of SGD
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.decreasing_temp = decreasing_temp
        self.temp_decay = temp_decay
        self.initial_temp = temperature

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        pass

    def train_episode(self, env):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        # Collect trajectory
        while not done:
            action, log_prob, _ = self.policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Calculate returns
        returns = self._compute_returns(rewards)

        # Update policy
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Decay temperature if using decreasing temperature
        if self.decreasing_temp:
            self.policy.temperature *= self.temp_decay

        return sum(rewards)

    def _compute_returns(self, rewards):
        # Calculate discounted return for each time step
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # Normalize returns for stable training
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def reset_temperature(self):
        self.policy.temperature = self.initial_temp


def run_trial(epsilon, step_size, seed, env, algorithm, use_buffer, env_name):
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = REINFORCE(env, step_size, epsilon, algorithm)

    max_steps_per_episode = 500

    torch.set_grad_enabled(True)

    episode_rewards = []
    for _ in range(1000):
        state, _ = env.reset()
        if env_name == "ALE/Assault-ram-v5":
            state = (state / 255) - 0.5
        done = False
        total_reward = 0
        n_steps = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            if env_name == "ALE/Assault-ram-v5":
                next_state = (next_state / 255) - 0.5
            done = done or truncated

            total_reward += reward

            agent.update(
                torch.as_tensor(state).to(agent.device).unsqueeze(0),
                torch.as_tensor(action).to(agent.device).unsqueeze(0),
                torch.as_tensor(reward).to(agent.device).unsqueeze(0),
                torch.as_tensor(next_state).to(agent.device).unsqueeze(0),
                torch.as_tensor(done).float().to(agent.device).unsqueeze(0),
            )

            state = next_state
            n_steps += 1
            if n_steps >= max_steps_per_episode:
                done = True

        episode_rewards.append(total_reward)

    return episode_rewards


# ACTOR-CRITIC IMPLEMENTATION
class ActorCritic:
    def __init__(
        self,
        state_dim,
        action_dim,
        temperature=1.0,
        actor_lr=0.001,
        critic_lr=0.001,
        gamma=0.99,
        decreasing_temp=False,
        temp_decay=0.999,
    ):
        self.policy_network = ZNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        self.policy = BoltzmannPolicy(self.policy_network, temperature)

        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.value_network.parameters(), lr=critic_lr
        )

        self.gamma = gamma
        self.decreasing_temp = decreasing_temp
        self.temp_decay = temp_decay
        self.initial_temp = temperature

    def train_episode(self, env):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        done = False

        # Collect trajectory
        while not done:
            value = self.value_network(state)
            action, log_prob, _ = self.policy.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward]))
            masks.append(torch.tensor([1 - done], dtype=torch.float))

            state = next_state

        # Calculate returns and advantages
        returns = self._compute_returns(rewards, masks)
        advantages = self._compute_advantages(returns, values)

        # Update policy and value networks
        actor_loss = 0
        critic_loss = 0

        for log_prob, advantage, value, R in zip(
            log_probs, advantages, values, returns
        ):
            actor_loss += -log_prob * advantage.detach()
            critic_loss += F.mse_loss(value, R)

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Decay temperature if using decreasing temperature
        if self.decreasing_temp:
            self.policy.temperature *= self.temp_decay

        return sum([r.item() for r in rewards])

    def _compute_returns(self, rewards, masks):
        returns = []
        R = 0

        for r, mask in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * mask
            returns.insert(0, R)

        return returns

    def _compute_advantages(self, returns, values):
        advantages = []
        for R, value in zip(returns, values):
            advantages.append(R - value.detach())

        return advantages

    def reset_temperature(self):
        self.policy.temperature = self.initial_temp


# TRAINING FUNCTION
def train_agents(env_name, num_seeds=50, num_episodes=1000, fixed_temp=1.0):
    # Initialize environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Results storage
    reinforce_fixed_rewards = np.zeros((num_seeds, num_episodes))
    reinforce_decay_rewards = np.zeros((num_seeds, num_episodes))
    ac_fixed_rewards = np.zeros((num_seeds, num_episodes))
    ac_decay_rewards = np.zeros((num_seeds, num_episodes))

    # Training loop for each seed
    for seed in tqdm(range(num_seeds), desc=f"Training on {env_name}"):
        set_seed(seed)

        # Initialize agents
        reinforce_fixed = REINFORCE(
            state_dim, action_dim, temperature=fixed_temp, decreasing_temp=False
        )
        reinforce_decay = REINFORCE(
            state_dim, action_dim, temperature=fixed_temp, decreasing_temp=True
        )
        ac_fixed = ActorCritic(
            state_dim, action_dim, temperature=fixed_temp, decreasing_temp=False
        )
        ac_decay = ActorCritic(
            state_dim, action_dim, temperature=fixed_temp, decreasing_temp=True
        )

        # Training for each episode
        for episode in range(num_episodes):
            # Train REINFORCE with fixed temperature
            reinforce_fixed_rewards[seed, episode] = reinforce_fixed.train_episode(env)

            # Train REINFORCE with decaying temperature
            reinforce_decay_rewards[seed, episode] = reinforce_decay.train_episode(env)

            # Train Actor-Critic with fixed temperature
            ac_fixed_rewards[seed, episode] = ac_fixed.train_episode(env)

            # Train Actor-Critic with decaying temperature
            ac_decay_rewards[seed, episode] = ac_decay.train_episode(env)

    # Close environment
    env.close()

    return {
        "reinforce_fixed": reinforce_fixed_rewards,
        "reinforce_decay": reinforce_decay_rewards,
        "ac_fixed": ac_fixed_rewards,
        "ac_decay": ac_decay_rewards,
    }


# PLOTTING FUNCTION
def plot_results(env_name, results, window_size=10):
    """
    Plot training curves with mean and standard deviation
    For each environment: 2 graphs (fixed temp and decreasing temp)
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Process data with smoothing
    def smooth_data(data, window_size):
        smoothed = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed[i] = np.array(
                [
                    np.mean(data[i, max(0, j - window_size) : j + 1])
                    for j in range(data.shape[1])
                ]
            )
        return smoothed

    reinforce_fixed_smoothed = smooth_data(results["reinforce_fixed"], window_size)
    reinforce_decay_smoothed = smooth_data(results["reinforce_decay"], window_size)
    ac_fixed_smoothed = smooth_data(results["ac_fixed"], window_size)
    ac_decay_smoothed = smooth_data(results["ac_decay"], window_size)

    # Fixed temperature plot
    ax = axes[0]
    episodes = np.arange(results["reinforce_fixed"].shape[1])

    # REINFORCE with fixed temperature (green, solid)
    mean_rewards = np.mean(reinforce_fixed_smoothed, axis=0)
    std_rewards = np.std(reinforce_fixed_smoothed, axis=0)
    ax.plot(
        episodes,
        mean_rewards,
        color="green",
        linestyle="-",
        label="REINFORCE (Fixed T)",
    )
    ax.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="green",
        alpha=0.2,
    )

    # Actor-Critic with fixed temperature (red, solid)
    mean_rewards = np.mean(ac_fixed_smoothed, axis=0)
    std_rewards = np.std(ac_fixed_smoothed, axis=0)
    ax.plot(
        episodes,
        mean_rewards,
        color="red",
        linestyle="-",
        label="Actor-Critic (Fixed T)",
    )
    ax.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="red",
        alpha=0.2,
    )

    ax.set_title(f"{env_name} - Fixed Temperature")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True)

    # Decreasing temperature plot
    ax = axes[1]

    # REINFORCE with decaying temperature (green, dashed)
    mean_rewards = np.mean(reinforce_decay_smoothed, axis=0)
    std_rewards = np.std(reinforce_decay_smoothed, axis=0)
    ax.plot(
        episodes,
        mean_rewards,
        color="green",
        linestyle="--",
        label="REINFORCE (Decaying T)",
    )
    ax.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="green",
        alpha=0.2,
    )

    # Actor-Critic with decaying temperature (red, dashed)
    mean_rewards = np.mean(ac_decay_smoothed, axis=0)
    std_rewards = np.std(ac_decay_smoothed, axis=0)
    ax.plot(
        episodes,
        mean_rewards,
        color="red",
        linestyle="--",
        label="Actor-Critic (Decaying T)",
    )
    ax.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="red",
        alpha=0.2,
    )

    ax.set_title(f"{env_name} - Decreasing Temperature")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{env_name}_results.png")
    plt.close()


# MAIN EXECUTION
if __name__ == "__main__":
    # Hyperparameters
    NUM_SEEDS = 50
    NUM_EPISODES = 1000
    FIXED_TEMP = 1.0  # You can adjust this value

    # Train on Acrobot-v1
    print("Training on Acrobot-v1...")
    acrobot_results = train_agents("Acrobot-v1", NUM_SEEDS, NUM_EPISODES, FIXED_TEMP)
    plot_results("Acrobot-v1", acrobot_results)

    # Train on ALE/Assault-ram-v5
    print("Training on ALE/Assault-ram-v5...")
    assault_results = train_agents(
        "ALE/Assault-ram-v5", NUM_SEEDS, NUM_EPISODES, FIXED_TEMP
    )
    plot_results("ALE/Assault-ram-v5", assault_results)

    print("Training complete! Results saved as PNG files.")
