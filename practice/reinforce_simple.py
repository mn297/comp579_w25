"""
REINFORCE Algorithm: Simple Implementation
- Policy Gradient method that directly optimizes the policy
- Uses complete episode returns to update policy parameters
- No baseline/critic (pure Monte Carlo policy gradient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# PART 1: POLICY NETWORK
class PolicyNetwork(nn.Module):
    """Simple policy network that outputs action probabilities"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Simple 2-layer network
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        # Convert input to tensor if it's not already
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Forward pass through network
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)

        # Convert to probabilities with softmax
        return F.softmax(action_logits, dim=-1)

    def select_action(self, state):
        """Sample action from the policy distribution"""
        probs = self.forward(state)
        # Create a categorical distribution and sample
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        # Return action and its log probability
        return action.item(), m.log_prob(action)


# PART 2: COLLECT EPISODE TRAJECTORY
def collect_trajectory(env, policy):
    """Run one episode and collect states, actions, rewards and log_probs"""
    state, _ = env.reset()

    # Lists to store episode history
    log_probs = []
    rewards = []
    states = []
    actions = []

    # Run episode until termination
    done = False
    while not done:
        # Select action from policy
        action, log_prob = policy.select_action(state)

        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store step information
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state

    return states, actions, rewards, log_probs


# PART 3: COMPUTE RETURNS
def compute_returns(rewards, gamma=0.99):
    """Calculate discounted returns for each timestep"""
    returns = []
    R = 0

    # Calculate returns from back to front (more efficient)
    for r in reversed(rewards):
        # R = r + gamma * R
        R = r + gamma * R
        # Insert at front to maintain chronological order
        returns.insert(0, R)

    # Convert to tensor and normalize (helps with training stability)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    return returns


# PART 4: POLICY GRADIENT UPDATE
def update_policy(optimizer, log_probs, returns):
    """Update policy parameters using REINFORCE gradient"""
    # Calculate loss: negative because we're maximizing expected return
    # REINFORCE objective: maximize E[R * log π(a|s)]
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)  # Negative for gradient ascent

    # Sum up all the losses
    policy_loss = torch.cat(policy_loss).sum()

    # Backpropagation
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item()


# PART 5: TRAINING LOOP
def train(env_name="CartPole-v1", num_episodes=500, gamma=0.99, lr=0.01):
    """Train policy using REINFORCE algorithm"""
    env = gym.make(env_name)

    # Get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create policy and optimizer
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training tracking
    episode_rewards = []

    for episode in range(num_episodes):
        # Collect trajectory
        _, _, rewards, log_probs = collect_trajectory(env, policy)
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Compute returns and update policy
        returns = compute_returns(rewards, gamma)
        loss = update_policy(optimizer, log_probs, returns)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode+1}: Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}"
            )

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE Learning Curve")
    plt.savefig("reinforce_learning_curve.png")
    plt.close()

    return policy, episode_rewards


# PART 6: MAIN EXECUTION
if __name__ == "__main__":
    # Train policy
    policy, rewards = train(num_episodes=300)
    print(f"Final average reward over last 10 episodes: {np.mean(rewards[-10:]):.2f}")

    # Visualize trained policy (optional)
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    done = False

    while not done:
        action, _ = policy.select_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

    env.close()
