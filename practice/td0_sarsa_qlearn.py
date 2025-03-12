"""
Implementation of Tabular TD(0) and SARSA Algorithms

This script provides implementations of:
1. Tabular TD(0) for policy evaluation (estimating V_π)
2. SARSA (on-policy TD control) for learning the optimal policy (Q-learning variant)
3. Q-Learning (off-policy TD control) for learning the optimal policy

Both algorithms are implemented with tabular representations (dictionaries),
making them suitable for small, discrete environments.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm


class TabularTD0:
    """
    Tabular TD(0) for policy evaluation (V function approximation).
    Implements the first algorithm from the image.
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, num_episodes=500):
        self.env = env
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.num_episodes = num_episodes

        # Initialize V(s) arbitrarily for all states
        self.V = defaultdict(float)  # Default value of 0

        # Performance tracking
        self.episode_rewards = []

    def policy(self, state):
        """
        The policy to be evaluated. This is a simple random policy.
        Can be replaced with any policy.
        """
        return random.randint(0, self.env.action_space.n - 1)

    def train(self):
        """Train the TD(0) algorithm to estimate V_π."""
        for episode in tqdm(range(self.num_episodes), desc="TD(0) Training"):
            # Initialize S
            state, _ = self.env.reset()
            state = self._discretize_state(state)
            done = False
            episode_reward = 0

            # Loop for each step of episode
            while not done:
                # Action given by π for state
                action = self.policy(state)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretize_state(next_state)
                done = terminated or truncated
                episode_reward += reward

                # TD Update: V(S) ← V(S) + α[R + γV(S') - V(S)]
                if done and terminated:  # Terminal state
                    target = reward  # V(terminal) = 0
                else:
                    target = reward + self.gamma * self.V[next_state]

                # Update rule
                self.V[state] += self.alpha * (target - self.V[state])

                # S ← S'
                state = next_state

            self.episode_rewards.append(episode_reward)

        return self.V, self.episode_rewards

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete for tabular methods.
        This is a simple discretization - for more complex environments,
        a more sophisticated approach would be needed.
        """
        # For CartPole, we can discretize the 4 continuous values
        if isinstance(state, np.ndarray) and len(state) == 4:
            # Cart position, cart velocity, pole angle, pole velocity
            cart_pos_bins = np.linspace(-2.4, 2.4, 10)
            cart_vel_bins = np.linspace(-4, 4, 10)
            pole_ang_bins = np.linspace(-0.2, 0.2, 10)
            pole_vel_bins = np.linspace(-3, 3, 10)

            # Get bin indices
            cart_pos_idx = np.digitize(state[0], cart_pos_bins)
            cart_vel_idx = np.digitize(state[1], cart_vel_bins)
            pole_ang_idx = np.digitize(state[2], pole_ang_bins)
            pole_vel_idx = np.digitize(state[3], pole_vel_bins)

            # Convert to tuple for dictionary key (hashable)
            return (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)

        # For other environments or discrete states, just return as is
        return tuple(state) if hasattr(state, "__iter__") else state


class SARSA:
    """
    SARSA (on-policy TD control) for Q-learning.
    Implements the second algorithm from the image.
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500):
        self.env = env
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # For ε-greedy policy
        self.num_episodes = num_episodes

        # Initialize Q(s,a) arbitrarily for all s ∈ S+, a ∈ A(s)
        self.Q = defaultdict(float)  # Default value of 0

        # Performance tracking
        self.episode_rewards = []

    def get_action(self, state):
        """
        Choose action using ε-greedy policy derived from Q.
        With probability ε, choose a random action.
        With probability 1-ε, choose the greedy action.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: choose random action
            return random.randint(0, self.env.action_space.n - 1)
        else:
            # Exploitation: choose best action based on Q values
            q_values = [self.Q[(state, a)] for a in range(self.env.action_space.n)]
            return np.argmax(q_values)

    def train(self):
        """Train the SARSA algorithm to learn Q."""
        for episode in tqdm(range(self.num_episodes), desc="SARSA Training"):
            # Initialize S
            state, _ = self.env.reset()
            state = self._discretize_state(state)

            # Choose A from S using policy derived from Q (e.g., ε-greedy)
            action = self.get_action(state)

            done = False
            episode_reward = 0

            # Loop for each step of episode
            while not done:
                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretize_state(next_state)
                done = terminated or truncated
                episode_reward += reward

                # Choose A' from S' using policy derived from Q (e.g., ε-greedy)
                next_action = self.get_action(next_state)

                # SARSA Update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                if done and terminated:  # Terminal state
                    target = reward  # Q(terminal, ·) = 0
                else:
                    target = reward + self.gamma * self.Q[(next_state, next_action)]

                # Update rule
                self.Q[(state, action)] += self.alpha * (
                    target - self.Q[(state, action)]
                )

                # S ← S'; A ← A'
                state = next_state
                action = next_action

            self.episode_rewards.append(episode_reward)

        return self.Q, self.episode_rewards

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete for tabular methods.
        This is a simple discretization - for more complex environments,
        a more sophisticated approach would be needed.
        """
        # For CartPole, we can discretize the 4 continuous values
        if isinstance(state, np.ndarray) and len(state) == 4:
            # Cart position, cart velocity, pole angle, pole velocity
            cart_pos_bins = np.linspace(-2.4, 2.4, 10)
            cart_vel_bins = np.linspace(-4, 4, 10)
            pole_ang_bins = np.linspace(-0.2, 0.2, 10)
            pole_vel_bins = np.linspace(-3, 3, 10)

            # Get bin indices
            cart_pos_idx = np.digitize(state[0], cart_pos_bins)
            cart_vel_idx = np.digitize(state[1], cart_vel_bins)
            pole_ang_idx = np.digitize(state[2], pole_ang_bins)
            pole_vel_idx = np.digitize(state[3], pole_vel_bins)

            # Convert to tuple for dictionary key (hashable)
            return (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)

        # For other environments or discrete states, just return as is
        return tuple(state) if hasattr(state, "__iter__") else state

    def evaluate_policy(self, num_episodes=10):
        """Evaluate the learned policy."""
        total_rewards = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = self._discretize_state(state)
            done = False
            episode_reward = 0

            while not done:
                # Always choose the greedy action (exploit only)
                q_values = [self.Q[(state, a)] for a in range(self.env.action_space.n)]
                action = np.argmax(q_values)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretize_state(next_state)
                done = terminated or truncated
                episode_reward += reward

                state = next_state

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)


class QLearning:
    """
    Q-Learning (off-policy TD control) for learning optimal policy.
    Implements the algorithm from the image showing off-policy TD control.
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500):
        self.env = env
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # For ε-greedy policy
        self.num_episodes = num_episodes

        # Initialize Q(s,a) arbitrarily for all s ∈ S+, a ∈ A(s)
        self.Q = defaultdict(float)  # Default value of 0

        # Performance tracking
        self.episode_rewards = []

    def get_action(self, state):
        """
        Choose action using ε-greedy policy derived from Q.
        With probability ε, choose a random action.
        With probability 1-ε, choose the greedy action.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: choose random action
            return random.randint(0, self.env.action_space.n - 1)
        else:
            # Exploitation: choose best action based on Q values
            q_values = [self.Q[(state, a)] for a in range(self.env.action_space.n)]
            return np.argmax(q_values)

    def train(self):
        """Train the Q-Learning algorithm to learn Q."""
        for episode in tqdm(range(self.num_episodes), desc="Q-Learning Training"):
            # Initialize S
            state, _ = self.env.reset()
            state = self._discretize_state(state)

            done = False
            episode_reward = 0

            # Loop for each step of episode
            while not done:
                # Choose A from S using policy derived from Q (e.g., ε-greedy)
                action = self.get_action(state)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretize_state(next_state)
                done = terminated or truncated
                episode_reward += reward

                # Q-Learning Update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                if done and terminated:  # Terminal state
                    target = reward  # Q(terminal, ·) = 0
                else:
                    # Find the maximum Q-value for the next state
                    next_q_values = [
                        self.Q[(next_state, a)] for a in range(self.env.action_space.n)
                    ]
                    max_next_q = max(next_q_values)
                    target = reward + self.gamma * max_next_q

                # Update rule
                self.Q[(state, action)] += self.alpha * (
                    target - self.Q[(state, action)]
                )

                # S ← S'
                state = next_state

            self.episode_rewards.append(episode_reward)

        return self.Q, self.episode_rewards

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete for tabular methods.
        This is a simple discretization - for more complex environments,
        a more sophisticated approach would be needed.
        """
        # For CartPole, we can discretize the 4 continuous values
        if isinstance(state, np.ndarray) and len(state) == 4:
            # Cart position, cart velocity, pole angle, pole velocity
            cart_pos_bins = np.linspace(-2.4, 2.4, 10)
            cart_vel_bins = np.linspace(-4, 4, 10)
            pole_ang_bins = np.linspace(-0.2, 0.2, 10)
            pole_vel_bins = np.linspace(-3, 3, 10)

            # Get bin indices
            cart_pos_idx = np.digitize(state[0], cart_pos_bins)
            cart_vel_idx = np.digitize(state[1], cart_vel_bins)
            pole_ang_idx = np.digitize(state[2], pole_ang_bins)
            pole_vel_idx = np.digitize(state[3], pole_vel_bins)

            # Convert to tuple for dictionary key (hashable)
            return (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)

        # For other environments or discrete states, just return as is
        return tuple(state) if hasattr(state, "__iter__") else state

    def evaluate_policy(self, num_episodes=10):
        """Evaluate the learned policy."""
        total_rewards = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = self._discretize_state(state)
            done = False
            episode_reward = 0

            while not done:
                # Always choose the greedy action (exploit only)
                q_values = [self.Q[(state, a)] for a in range(self.env.action_space.n)]
                action = np.argmax(q_values)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretize_state(next_state)
                done = terminated or truncated
                episode_reward += reward

                state = next_state

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)


def plot_learning_curves(td0_rewards, sarsa_rewards, qlearning_rewards=None):
    """Plot learning curves for all algorithms."""
    plt.figure(figsize=(15, 5))

    num_plots = 3 if qlearning_rewards is not None else 2

    # Plot TD(0) learning curve
    plt.subplot(1, num_plots, 1)
    plt.plot(td0_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("TD(0) Learning Curve")
    plt.grid(True)

    # Plot SARSA learning curve
    plt.subplot(1, num_plots, 2)
    plt.plot(sarsa_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA Learning Curve")
    plt.grid(True)

    # Plot Q-Learning curve if available
    if qlearning_rewards is not None:
        plt.subplot(1, num_plots, 3)
        plt.plot(qlearning_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-Learning Curve")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Create environment
    env = gym.make("CartPole-v1")

    # Train TD(0) for estimating V_π
    print("\nTraining TD(0) for estimating V_π...")
    td0 = TabularTD0(env, alpha=0.1, gamma=0.99, num_episodes=500)
    v_function, td0_rewards = td0.train()
    print(f"Average reward over last 100 episodes: {np.mean(td0_rewards[-100:]):.2f}")

    # Train SARSA for learning Q
    print("\nTraining SARSA for learning Q...")
    sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500)
    q_function_sarsa, sarsa_rewards = sarsa.train()
    print(f"Average reward over last 100 episodes: {np.mean(sarsa_rewards[-100:]):.2f}")

    # Train Q-Learning for learning Q
    print("\nTraining Q-Learning for learning Q...")
    qlearning = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500)
    q_function_ql, qlearning_rewards = qlearning.train()
    print(
        f"Average reward over last 100 episodes: {np.mean(qlearning_rewards[-100:]):.2f}"
    )

    # Evaluate policies
    sarsa_avg_reward = sarsa.evaluate_policy(num_episodes=10)
    qlearning_avg_reward = qlearning.evaluate_policy(num_episodes=10)
    print(f"\nSARSA policy evaluation - Average reward: {sarsa_avg_reward:.2f}")
    print(f"Q-Learning policy evaluation - Average reward: {qlearning_avg_reward:.2f}")

    # Plot learning curves for all three algorithms
    plot_learning_curves(td0_rewards, sarsa_rewards, qlearning_rewards)

    # Compare algorithms directly
    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_rewards, label="SARSA (On-policy)")
    plt.plot(qlearning_rewards, label="Q-Learning (Off-policy)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA vs. Q-Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Close the environment
    env.close()
