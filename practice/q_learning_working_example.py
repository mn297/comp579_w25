"""
Complete Working Implementation of Deep Q-Learning with PyTorch

This script provides a complete, runnable implementation of Deep Q-Learning
using PyTorch. It includes:

1. A neural network Q-function approximator
2. Experience replay buffer
3. Training loop with visualization
4. Both Q-Learning and Expected SARSA implementations

The example uses the CartPole-v1 environment, which is simple but demonstrates
all the key concepts of Deep Q-Learning.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm


# Neural network for Q-function approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(QNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),      # Smaller network for faster training
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        
        # Initialize weights with small random values
        self.mlp.apply(self.init_weights)
        
        self.device = device
        self.to(device)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)  # Slightly wider range for faster learning
            nn.init.uniform_(m.bias, -0.01, 0.01)
    
    def forward(self, x):
        return self.mlp(x)


# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        """Store a transition (s, a, r, s', done) in the buffer"""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        
        # Convert to tensors and move to the appropriate device
        state = torch.as_tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.as_tensor(np.array(action), dtype=torch.int64).to(self.device)
        reward = torch.as_tensor(np.array(reward), dtype=torch.float32).to(self.device)
        next_state = torch.as_tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.as_tensor(np.array(done), dtype=torch.float32).to(self.device)
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


# Deep Q-Learning agent
class DQNAgent:
    def __init__(self, env, learning_rate=0.001, epsilon=0.1, gamma=0.99, algorithm="Q-Learning"):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.algorithm = algorithm
        
        # Get environment dimensions
        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        
        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create Q-network and optimizer
        self.q_network = QNetwork(self.state_dim, self.n_actions, self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(10000, self.device)
        self.batch_size = 64
        
        # For tracking progress
        self.rewards_history = []
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.uniform() < self.epsilon:
            # Exploration: select random action
            return np.random.choice(self.n_actions)
        else:
            # Exploitation: select best action according to Q-network
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()
    
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """Update Q-network using a batch of experiences"""
        # Get current Q-values for the actions that were taken
        q_values = self.q_network(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            # For terminal states, target is just the reward
            # For non-terminal states, target includes discounted future rewards
            next_q_values = self.q_network(next_state_batch)
            
            if self.algorithm == "Q-Learning":
                # Q-Learning: Use maximum Q-value in next state
                best_next_q_values, _ = next_q_values.max(dim=1)
                targets = reward_batch + (1 - done_batch) * self.gamma * best_next_q_values
            else:
                # Expected SARSA: Use expected Q-value under epsilon-greedy policy
                best_next_q_values, _ = next_q_values.max(dim=1)
                avg_next_q_values = next_q_values.mean(dim=1)
                expected_next_q_values = (
                    self.epsilon * avg_next_q_values +
                    (1 - self.epsilon) * best_next_q_values
                )
                targets = reward_batch + (1 - done_batch) * self.gamma * expected_next_q_values
        
        # Compute loss and update weights
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=200, max_steps=500):
        """Train the agent for a specified number of episodes"""
        all_rewards = []
        avg_rewards = []
        
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            # Experience collection
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.replay_buffer.push((state, action, reward, next_state, done))
                
                # Learn from experience
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.update(*batch)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Track progress
            all_rewards.append(episode_reward)
            avg_reward = np.mean(all_rewards[-100:])  # Average over last 100 episodes
            avg_rewards.append(avg_reward)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
            
        self.rewards_history = all_rewards
        return all_rewards, avg_rewards
    
    def plot_rewards(self):
        """Plot the rewards obtained during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history)
        plt.plot(np.convolve(self.rewards_history, np.ones(10)/10, mode='valid'))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Rewards over Episodes ({self.algorithm})')
        plt.grid(True)
        plt.show()
    
    def save(self, filename):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filename}")
    
    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the agent's performance"""
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Always select best action during evaluation
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action = torch.argmax(q_values).item()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
                
                if render:
                    self.env.render()
            
            total_rewards.append(episode_reward)
            print(f"Evaluation episode {episode+1}/{num_episodes}, Reward: {episode_reward}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Average evaluation reward: {avg_reward:.2f}")
        return avg_reward


# Main execution
if __name__ == "__main__":
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create and train agent
    agent = DQNAgent(
        env=env,
        learning_rate=0.001,
        epsilon=0.1,
        gamma=0.99,
        algorithm="Q-Learning"  # Change to "ExpectedSARSA" to use that algorithm
    )
    
    # Train the agent
    rewards, avg_rewards = agent.train(num_episodes=200)
    
    # Plot training progress
    agent.plot_rewards()
    
    # Save the trained model
    agent.save("cartpole_dqn.pt")
    
    # Evaluate the trained agent
    avg_eval_reward = agent.evaluate(num_episodes=10)
    
    # Compare with random policy
    random_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        random_rewards.append(episode_reward)
    
    print(f"Random policy average reward: {np.mean(random_rewards):.2f}")
    print(f"DQN policy average reward: {avg_eval_reward:.2f}")
    
    # Close the environment
    env.close()
