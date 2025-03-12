"""
Bite-sized Prioritized Experience Replay Example

This example demonstrates a simple implementation of Prioritized Experience Replay (PER)
for Deep Q-Learning. PER samples transitions with higher TD error more frequently,
focusing learning on the most "surprising" or "informative" experiences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt


# Neural network for Q-function approximation
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)


# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01
    ):
        """
        Initialize a prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta: Controls importance sampling weights (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time we sample
            epsilon: Small constant added to TD errors to ensure non-zero probability
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initialize with max priority for new experiences

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer with max priority"""
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        # New experiences get maximum priority to ensure they're sampled
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities"""
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[: len(self.buffer)]
        else:
            priorities = self.priorities

        # Convert priorities to probabilities (higher priority = higher probability)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on the probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Get the experiences for these indices
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Calculate importance sampling weights to correct for bias
        # (because we're sampling non-uniformly)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Increment beta for next sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,  # Need indices to update priorities later
            weights,  # Importance sampling weights
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors"""
        for i, td_error in zip(indices, td_errors):
            # Add epsilon for stability and to ensure non-zero probability
            priority = abs(td_error) + self.epsilon
            self.priorities[i] = priority
            # Update max priority for use with new experiences
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# Agent with Prioritized Experience Replay
class DQNAgentWithPER:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        # Environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        self.batch_size = 64

        # For tracking progress
        self.rewards = []

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    def update(self):
        """Learn from a batch of experiences using prioritized replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0  # Not enough experiences yet

        # Sample a batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(
            self.device
        )  # Importance sampling weights

        # Get current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (no gradient tracking needed)
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            best_next_q_values, _ = next_q_values.max(dim=1)
            targets = rewards + (1 - dones) * self.gamma * best_next_q_values

        # Compute TD errors (for updating priorities)
        td_errors = targets - q_values

        # Update replay buffer priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Compute loss - weighted by importance sampling
        # We use Huber loss for stability (less sensitive to outliers than MSE)
        loss_fn = nn.SmoothL1Loss(
            reduction="none"
        )  # Don't reduce, we need per-sample loss
        elementwise_loss = loss_fn(q_values, targets)
        weighted_loss = (
            elementwise_loss * weights
        ).mean()  # Apply weights and then reduce

        # Update weights
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return weighted_loss.item()

    def train(self, num_episodes=200):
        """Train the agent for a specified number of episodes"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store experience in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Update network
                loss = self.update()

                # Move to next state
                state = next_state
                episode_reward += reward

            self.rewards.append(episode_reward)

            # Print progress periodically
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.rewards[-10:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")

        return self.rewards


# Utility to plot learning curves
def plot_rewards(standard_rewards, prioritized_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(standard_rewards, label="Standard Replay")
    plt.plot(prioritized_rewards, label="Prioritized Replay")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Comparison: Standard vs Prioritized Experience Replay")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Create environment
    env = gym.make("CartPole-v1")

    print("Training with Standard Experience Replay...")
    # Create a version with standard replay for comparison
    from deep_q import DQNAgent  # Import from your existing module

    standard_agent = DQNAgent(env, learning_rate=0.001, epsilon=0.1, gamma=0.99)
    standard_rewards, _ = standard_agent.train(num_episodes=100)

    print("\nTraining with Prioritized Experience Replay...")
    # Create and train agent with prioritized replay
    per_agent = DQNAgentWithPER(env, learning_rate=0.001, epsilon=0.1, gamma=0.99)
    prioritized_rewards = per_agent.train(num_episodes=100)

    # Plot comparison
    plot_rewards(standard_rewards, prioritized_rewards)

    # Close environment
    env.close()
