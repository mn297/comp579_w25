import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import clear_output
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Set up the environment
def create_environment(map_size="4x4", is_slippery=True):
    """Create a FrozenLake environment with specified parameters."""
    if map_size == "4x4":
        return gym.make('FrozenLake-v1', render_mode=None, is_slippery=is_slippery)
    elif map_size == "8x8":
        return gym.make('FrozenLake8x8-v1', render_mode=None, is_slippery=is_slippery)
    else:
        raise ValueError("map_size must be either '4x4' or '8x8'")

# Visualization functions
def plot_policy_and_value(env, policy, value_function=None, title="Policy"):
    """Visualize the policy and optionally the value function."""
    if hasattr(env, 'desc'):
        map_desc = env.desc
    else:
        # For environments without a desc attribute
        if env.observation_space.n == 16:  # 4x4 grid
            map_desc = np.array([
                ['S', 'F', 'F', 'F'],
                ['F', 'H', 'F', 'H'],
                ['F', 'F', 'F', 'H'],
                ['H', 'F', 'F', 'G']
            ])
        elif env.observation_space.n == 64:  # 8x8 grid
            # This is an approximation, actual 8x8 map may differ
            map_desc = np.array([
                ['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
                ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                ['F', 'H', 'H', 'F', 'F', 'F', 'H', 'F'],
                ['F', 'H', 'F', 'F', 'H', 'F', 'H', 'F'],
                ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'G']
            ])
    
    grid_size = len(map_desc)
    
    # Create a figure with one or two subplots
    if value_function is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Define colors for the grid
    colors = {
        b'S': 'lightblue',  # Start
        b'F': 'white',      # Frozen
        b'H': 'black',      # Hole
        b'G': 'green'       # Goal
    }
    
    # Create a grid for the background
    grid_background = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if map_desc[i][j] == b'S' or map_desc[i][j] == 'S':
                grid_background[i, j] = 0  # Start
            elif map_desc[i][j] == b'F' or map_desc[i][j] == 'F':
                grid_background[i, j] = 1  # Frozen
            elif map_desc[i][j] == b'H' or map_desc[i][j] == 'H':
                grid_background[i, j] = 2  # Hole
            elif map_desc[i][j] == b'G' or map_desc[i][j] == 'G':
                grid_background[i, j] = 3  # Goal
    
    # Create a custom colormap
    cmap = ListedColormap(['lightblue', 'white', 'black', 'green'])
    
    # Plot the grid background
    ax1.imshow(grid_background, cmap=cmap)
    
    # Add arrows to show the policy
    for s in range(env.observation_space.n):
        a = policy[s]
        row, col = s // grid_size, s % grid_size
        
        # Skip holes and goal
        if map_desc[row][col] in [b'H', 'H', b'G', 'G']:
            continue
        
        # Define arrow directions
        if a == 0:  # Left
            dx, dy = -0.2, 0
            arrow = '←'
        elif a == 1:  # Down
            dx, dy = 0, 0.2
            arrow = '↓'
        elif a == 2:  # Right
            dx, dy = 0.2, 0
            arrow = '→'
        elif a == 3:  # Up
            dx, dy = 0, -0.2
            arrow = '↑'
        
        ax1.text(col, row, arrow, ha='center', va='center', fontsize=15, fontweight='bold')
    
    # Add grid lines
    ax1.grid(color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(np.arange(-0.5, grid_size, 1))
    ax1.set_yticks(np.arange(-0.5, grid_size, 1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(f"{title} - Policy")
    
    # Create legend
    patches = [
        mpatches.Patch(color='lightblue', label='Start'),
        mpatches.Patch(color='white', label='Frozen'),
        mpatches.Patch(color='black', label='Hole'),
        mpatches.Patch(color='green', label='Goal')
    ]
    ax1.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot value function if provided
    if value_function is not None:
        value_grid = value_function.reshape(grid_size, grid_size)
        im = ax2.imshow(value_grid, cmap='viridis')
        
        # Add value text
        for i in range(grid_size):
            for j in range(grid_size):
                # Skip holes
                if map_desc[i][j] in [b'H', 'H']:
                    continue
                ax2.text(j, i, f"{value_grid[i, j]:.2f}", ha='center', va='center', 
                         color='white' if value_grid[i, j] < 0.5 else 'black')
        
        ax2.grid(color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(np.arange(-0.5, grid_size, 1))
        ax2.set_yticks(np.arange(-0.5, grid_size, 1))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_title(f"{title} - Value Function")
        plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(rewards, window=100, title="Learning Curve"):
    """Plot the learning curve showing the moving average of rewards."""
    plt.figure(figsize=(10, 6))
    
    # Calculate moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Average (window={window})')
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_episode_with_rendering(env, policy, max_steps=100, delay=0.5):
    """Run a single episode using the given policy and render each step."""
    env_render = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)
    state, _ = env_render.reset()
    
    total_reward = 0
    done = False
    truncated = False
    step = 0
    
    while not (done or truncated) and step < max_steps:
        action = policy[state]
        state, reward, done, truncated, _ = env_render.step(action)
        total_reward += reward
        step += 1
        time.sleep(delay)  # Add delay for better visualization
    
    env_render.close()
    return total_reward

# Reinforcement Learning Algorithms

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """
    Value Iteration algorithm to find the optimal policy.
    
    Args:
        env: The environment
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
        
    Returns:
        policy: The optimal policy
        V: The optimal value function
    """
    # Initialize value function
    V = np.zeros(env.observation_space.n)
    
    for i in range(max_iterations):
        delta = 0
        
        for s in range(env.observation_space.n):
            v = V[s]
            
            # Calculate the value of each action
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
            
            # Update value function with the maximum action value
            V[s] = max(action_values)
            
            # Calculate delta for convergence check
            delta = max(delta, abs(v - V[s]))
        
        # Check for convergence
        if delta < theta:
            print(f"Value Iteration converged after {i+1} iterations.")
            break
    
    # Extract policy from value function
    policy = np.zeros(env.observation_space.n, dtype=int)
    
    for s in range(env.observation_space.n):
        action_values = np.zeros(env.action_space.n)
        
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
        
        policy[s] = np.argmax(action_values)
    
    return policy, V

def sarsa(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1, decay_rate=0.999):
    """
    SARSA algorithm (on-policy TD control).
    
    Args:
        env: The environment
        episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        decay_rate: Rate at which epsilon decays
        
    Returns:
        policy: The learned policy
        Q: The learned action-value function
        rewards: List of rewards per episode
    """
    # Initialize Q-table
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Initialize rewards list
    rewards = []
    
    # Epsilon-greedy policy
    def get_action(state, eps):
        if np.random.random() < eps:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])
    
    for episode in range(episodes):
        state, _ = env.reset()
        
        # Get first action using epsilon-greedy
        action = get_action(state, epsilon)
        
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Take action and observe next state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Get next action using epsilon-greedy
            next_action = get_action(next_state, epsilon)
            
            # SARSA update
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            # Move to next state and action
            state = next_state
            action = next_action
        
        # Decay epsilon
        epsilon *= decay_rate
        
        # Store reward
        rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(rewards[-100:]):.4f}, Epsilon: {epsilon:.4f}")
    
    # Extract policy from Q-table
    policy = np.argmax(Q, axis=1)
    
    return policy, Q, rewards

def expected_sarsa(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1, decay_rate=0.999):
    """
    Expected SARSA algorithm.
    
    Args:
        env: The environment
        episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        decay_rate: Rate at which epsilon decays
        
    Returns:
        policy: The learned policy
        Q: The learned action-value function
        rewards: List of rewards per episode
    """
    # Initialize Q-table
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Initialize rewards list
    rewards = []
    
    # Epsilon-greedy policy
    def get_action(state, eps):
        if np.random.random() < eps:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])
    
    for episode in range(episodes):
        state, _ = env.reset()
        
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Get action using epsilon-greedy
            action = get_action(state, epsilon)
            
            # Take action and observe next state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Calculate expected value of next state
            expected_value = 0
            best_action = np.argmax(Q[next_state])
            
            # Probability of random action
            prob_random = epsilon / env.action_space.n
            
            # Probability of greedy action
            prob_greedy = 1 - epsilon + prob_random
            
            # Calculate expected value
            for a in range(env.action_space.n):
                if a == best_action:
                    expected_value += prob_greedy * Q[next_state, a]
                else:
                    expected_value += prob_random * Q[next_state, a]
            
            # Expected SARSA update
            Q[state, action] += alpha * (reward + gamma * expected_value - Q[state, action])
            
            # Move to next state
            state = next_state
        
        # Decay epsilon
        epsilon *= decay_rate
        
        # Store reward
        rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(rewards[-100:]):.4f}, Epsilon: {epsilon:.4f}")
    
    # Extract policy from Q-table
    policy = np.argmax(Q, axis=1)
    
    return policy, Q, rewards

def monte_carlo(env, episodes=10000, gamma=0.99, epsilon=0.1, decay_rate=0.999):
    """
    Monte Carlo control with epsilon-greedy policy.
    
    Args:
        env: The environment
        episodes: Number of episodes
        gamma: Discount factor
        epsilon: Exploration rate
        decay_rate: Rate at which epsilon decays
        
    Returns:
        policy: The learned policy
        Q: The learned action-value function
        rewards: List of rewards per episode
    """
    # Initialize Q-table and counts
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns_count = np.zeros((env.observation_space.n, env.action_space.n))
    returns_sum = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Initialize rewards list
    rewards = []
    
    # Epsilon-greedy policy
    def get_action(state, eps):
        if np.random.random() < eps:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])
    
    for episode in range(episodes):
        # Generate an episode
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = get_action(state, epsilon)
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            total_reward += reward
            state = next_state
        
        # Store total reward
        rewards.append(total_reward)
        
        # Calculate returns and update Q-values
        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            
            G = gamma * G + reward
            
            # First-visit MC: only update if this is the first occurrence of (s,a)
            if (state, action) not in [(episode_states[i], episode_actions[i]) for i in range(t)]:
                returns_count[state, action] += 1
                returns_sum[state, action] += G
                Q[state, action] = returns_sum[state, action] / returns_count[state, action]
        
        # Decay epsilon
        epsilon *= decay_rate
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(rewards[-100:]):.4f}, Epsilon: {epsilon:.4f}")
    
    # Extract policy from Q-table
    policy = np.argmax(Q, axis=1)
    
    return policy, Q, rewards

# Main function to run and compare algorithms
def main():
    # Create environment
    env = create_environment(map_size="4x4", is_slippery=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    env.reset(seed=42)
    
    print("Environment: FrozenLake-v1")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Choose algorithm to run
    print("\nAvailable algorithms:")
    print("1. Value Iteration")
    print("2. SARSA")
    print("3. Expected SARSA")
    print("4. Monte Carlo")
    
    choice = input("\nEnter algorithm number (1-4) or 'all' to run all: ")
    
    if choice == '1' or choice.lower() == 'all':
        print("\nRunning Value Iteration...")
        vi_policy, vi_value = value_iteration(env, gamma=0.99)
        plot_policy_and_value(env, vi_policy, vi_value, "Value Iteration")
        
        # Run an episode with the learned policy
        print("Running an episode with Value Iteration policy...")
        reward = run_episode_with_rendering(env, vi_policy)
        print(f"Episode reward: {reward}")
    
    if choice == '2' or choice.lower() == 'all':
        print("\nRunning SARSA...")
        sarsa_policy, sarsa_q, sarsa_rewards = sarsa(env, episodes=10000)
        
        # Extract value function from Q-table
        sarsa_value = np.max(sarsa_q, axis=1)
        
        plot_policy_and_value(env, sarsa_policy, sarsa_value, "SARSA")
        plot_learning_curve(sarsa_rewards, title="SARSA Learning Curve")
        
        # Run an episode with the learned policy
        print("Running an episode with SARSA policy...")
        reward = run_episode_with_rendering(env, sarsa_policy)
        print(f"Episode reward: {reward}")
    
    if choice == '3' or choice.lower() == 'all':
        print("\nRunning Expected SARSA...")
        esarsa_policy, esarsa_q, esarsa_rewards = expected_sarsa(env, episodes=10000)
        
        # Extract value function from Q-table
        esarsa_value = np.max(esarsa_q, axis=1)
        
        plot_policy_and_value(env, esarsa_policy, esarsa_value, "Expected SARSA")
        plot_learning_curve(esarsa_rewards, title="Expected SARSA Learning Curve")
        
        # Run an episode with the learned policy
        print("Running an episode with Expected SARSA policy...")
        reward = run_episode_with_rendering(env, esarsa_policy)
        print(f"Episode reward: {reward}")
    
    if choice == '4' or choice.lower() == 'all':
        print("\nRunning Monte Carlo...")
        mc_policy, mc_q, mc_rewards = monte_carlo(env, episodes=10000)
        
        # Extract value function from Q-table
        mc_value = np.max(mc_q, axis=1)
        
        plot_policy_and_value(env, mc_policy, mc_value, "Monte Carlo")
        plot_learning_curve(mc_rewards, title="Monte Carlo Learning Curve")
        
        # Run an episode with the learned policy
        print("Running an episode with Monte Carlo policy...")
        reward = run_episode_with_rendering(env, mc_policy)
        print(f"Episode reward: {reward}")
    
    # Compare algorithms if all were run
    if choice.lower() == 'all':
        # Compare learning curves
        plt.figure(figsize=(12, 6))
        
        window = 100
        if len(sarsa_rewards) >= window:
            plt.plot(range(window-1, len(sarsa_rewards)), 
                     np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid'), 
                     label='SARSA')
        
        if len(esarsa_rewards) >= window:
            plt.plot(range(window-1, len(esarsa_rewards)), 
                     np.convolve(esarsa_rewards, np.ones(window)/window, mode='valid'), 
                     label='Expected SARSA')
        
        if len(mc_rewards) >= window:
            plt.plot(range(window-1, len(mc_rewards)), 
                     np.convolve(mc_rewards, np.ones(window)/window, mode='valid'), 
                     label='Monte Carlo')
        
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Comparison of Learning Algorithms')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main() 