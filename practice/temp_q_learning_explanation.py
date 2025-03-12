"""
# Q-Learning in PyTorch: A Bite-Sized Explanation

This file contains explanations of the Q-Learning algorithm as implemented in the optimized version.
You can copy these code cells into your Jupyter notebook.
"""

# Cell 1 - Introduction to Q-Learning
"""
## Q-Learning: Core Concepts

Q-Learning is a reinforcement learning algorithm that learns to make optimal decisions by estimating the value (Q-value) of taking actions in states.

Key components:
1. **Q-values**: The expected future reward of taking action a in state s
2. **Epsilon-greedy policy**: Balance between exploration and exploitation
3. **Bellman equation**: Update rule for Q-values

In deep Q-learning, we use neural networks to approximate the Q-function.
"""

# Cell 2 - Neural Network for Q-Values
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import random

gym.register_envs(ale_py)


# This is the neural network that approximates Q-values
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(QNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),  # Input layer -> hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(256, 256),  # Hidden layer -> hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(256, output_dim),  # Hidden layer -> output layer (Q-values)
        )

        # Initialize weights with small random values
        self.mlp.apply(self.init_weights)

        self.device = device
        self.to(device)  # Move network to the specified device (CPU/GPU)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Small initialization helps with training stability
            nn.init.uniform_(m.weight, -0.001, 0.001)
            nn.init.uniform_(m.bias, -0.001, 0.001)

    def forward(self, x):
        return self.mlp(x)  # Output Q-values for all actions


# Cell 3 - Action Selection with Epsilon-Greedy Policy
"""
## Action Selection: Epsilon-Greedy Strategy

The epsilon-greedy strategy balances:
- **Exploration**: Try random actions (with probability ε)
- **Exploitation**: Choose the best known action (with probability 1-ε)
"""


def select_action(state, epsilon, q_network, n_actions, device):
    # Exploration: choose random action with probability epsilon
    if np.random.uniform() < epsilon:
        return np.random.choice(n_actions)

    # Exploitation: choose best action according to Q-network
    else:
        # Convert state to tensor and process through network
        state_tensor = torch.as_tensor(state).float().unsqueeze(0).to(device)

        # Get Q-values and select action with highest value
        with torch.no_grad():  # No need to track gradients for inference
            q_values = q_network(state_tensor)
            best_action = torch.argmax(q_values).item()

        return best_action


# Cell 4 - Q-Learning Update Rule
"""
## Q-Learning Update: The Core Algorithm

The Q-learning update uses the Bellman equation:
Q(s,a) ← Q(s,a) + α [r + γ * max_a'(Q(s',a')) - Q(s,a)]

Where:
- α: Learning rate
- γ: Discount factor
- r: Reward
- s': Next state
- max_a'(Q(s',a')): Maximum Q-value in the next state
"""


def update_q_network(
    q_network,
    optimizer,
    loss_fn,
    state_batch,
    action_batch,
    reward_batch,
    next_state_batch,
    done_batch,
    gamma,
):

    # Step 1: Get current Q-values for the actions that were taken
    q_values = q_network(state_batch)
    q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Step 2: Compute target Q-values using the Bellman equation
    with torch.no_grad():  # Don't compute gradients for target computation
        # For terminal states, target is just the reward
        # For non-terminal states, target includes discounted future rewards

        # Convert done flags to mask (1.0 for non-terminal, 0.0 for terminal)
        non_terminal_mask = 1.0 - done_batch

        # Get max Q-values for next states
        next_q_values = q_network(next_state_batch)
        max_next_q_values, _ = next_q_values.max(dim=1)

        # Q-learning target: r + γ * max_a'(Q(s',a'))
        target_q_values = reward_batch + non_terminal_mask * gamma * max_next_q_values

    # Step 3: Compute loss (how far current Q-values are from targets)
    loss = loss_fn(q_values, target_q_values)

    # Step 4: Update the network weights
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights


# Cell 5 - The Expected SARSA Variant
"""
## Expected SARSA: An Alternative to Q-Learning

While Q-learning uses max(Q) for the next state, Expected SARSA uses the expected Q-value:
- It averages over all possible actions in the next state
- This can be more stable in some environments

The target equation becomes:
Q(s,a) ← Q(s,a) + α [r + γ * Σ(π(a'|s') * Q(s',a')) - Q(s,a)]

Where π(a'|s') is the probability of taking action a' in state s'.
"""


def compute_expected_sarsa_target(
    next_q_values, epsilon, gamma, reward_batch, non_terminal_mask
):
    # Get best action values for next states
    greedy_next_q_values, _ = next_q_values.max(dim=1)

    # Average Q-value across all actions (for random policy component)
    random_next_q_values = next_q_values.mean(dim=1)

    # Expected SARSA combines random and greedy policies according to epsilon
    # With probability epsilon: random action
    # With probability (1-epsilon): greedy action
    expected_next_q_values = (
        epsilon * random_next_q_values + (1 - epsilon) * greedy_next_q_values
    )

    # Final target: r + γ * E[Q(s',a')]
    target_q_values = reward_batch + non_terminal_mask * gamma * expected_next_q_values

    return target_q_values


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """
        Store a transition (s, a, r, s', done) in the buffer

        In the optimized version, transitions are converted to tensors and
        moved to the appropriate device (CPU/GPU) before storage
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions

        Returns a batch of transitions formatted for efficient learning
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Check if GPU is available
use_gpu = True
if torch.cuda.is_available() and use_gpu:
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    if use_gpu:
        print("GPU requested but not available. Using CPU instead.")
    else:
        print("Using CPU as requested.")


# Initialize Q-network and environment
acrobot_env_name = "Acrobot-v1"
acrobot_env = gym.make(acrobot_env_name)
# acrobot_env = StepAPICompatibility(acrobot_env)
print("Action space:", acrobot_env.action_space)
print("State space:", acrobot_env.observation_space)
learning_rate = 0.001
num_episodes = 1000
epsilon = 0.1
batch_size = 64
gamma = 0.99
replay_buffer = ReplayBuffer(10000, device)


q_network = QNetwork(
    acrobot_env.observation_space.shape[0], acrobot_env.action_space.n, device
)

optimizer = torch.optim.SGD(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

for episode in range(num_episodes):
    state, _ = acrobot_env.reset()
    done = False

    while not done:
        # Select action using epsilon-greedy
        action = select_action(
            state, epsilon, q_network, acrobot_env.action_space.n, device
        )

        # Take action in environment
        next_state, reward, done, _, _ = acrobot_env.step(action)

        # Store experience in replay buffer
        replay_buffer.push((state, action, reward, next_state, done))

        # Sample batch from replay buffer and update Q-network
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
                batch
            )

            # Convert to tensors
            state_batch = torch.as_tensor(state_batch).float().to(device)
            action_batch = torch.as_tensor(action_batch).long().to(device)
            reward_batch = torch.as_tensor(reward_batch).float().to(device)
            next_state_batch = torch.as_tensor(next_state_batch).float().to(device)
            done_batch = torch.as_tensor(done_batch).float().to(device)

            # Update Q-network
            update_q_network(
                q_network,
                optimizer,
                loss_fn,
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch,
                gamma,
            )

        state = next_state


# Cell 10 - Analyzing the Update Method
"""
## Deep Dive: The Q-Network Update Method

The update method is the core of Q-learning:

```python
def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
    # 1. Calculate current Q-values
    q_val_batch = self.Q(state_batch)
    q_val_batch = q_val_batch.gather(1, action_batch.unsqueeze(1))
    q_val_batch = q_val_batch.squeeze(1)

    # 2. Calculate target Q-values (no gradient tracking needed)
    with torch.no_grad():
        done_batch = 1.0 - done_batch  # Convert to mask (1.0 for non-terminal)
        next_q_val = self.Q(next_state_batch)  # Q-values for all actions in next state
        
        # For Q-Learning: use maximum Q-value in next state
        greedy_next_q_val, _ = next_q_val.max(dim=1)
        
        if self.algorithm == "Q-Learning":
            # Q-Learning target: r + γ * max_a'(Q(s',a'))
            target_batch = reward_batch + done_batch * self.gamma * greedy_next_q_val
        else:
            # Expected SARSA: use expected Q-value across all actions
            random_next_q_val = next_q_val.mean(dim=1)
            exp_next_q_val = (
                self.epsilon * random_next_q_val
                + (1 - self.epsilon) * greedy_next_q_val
            )
            target_batch = reward_batch + done_batch * self.gamma * exp_next_q_val

    # 3. Calculate loss (Mean Squared Error between current and target Q-values)
    loss = self.loss_fn(q_val_batch, target_batch)
    
    # 4. Update network weights using backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

**Key PyTorch operations**:
- `gather()`: Selects Q-values for the actions that were taken
- `max(dim=1)`: Finds maximum Q-value for each batch entry along action dimension
- `mean(dim=1)`: Averages Q-values across all actions
- `torch.no_grad()`: Prevents gradient tracking during target calculation
"""


# Cell 11 - Advanced Topics in Deep Q-Learning
"""
## Advanced Topics in Deep Q-Learning

1. **Double Q-Learning**:
   - Problem: Standard Q-learning tends to overestimate Q-values
   - Solution: Use one network to select actions, another to evaluate them
   - Implementation: 
     ```python
     # Current network selects action
     best_actions = current_Q(next_states).argmax(dim=1)
     
     # Target network evaluates action
     next_q_values = target_Q(next_states).gather(1, best_actions.unsqueeze(1))
     ```

2. **Dueling Networks**:
   - Separates state value (V) and action advantage (A)
   - Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
   - Better at evaluating states where actions don't matter much

3. **Prioritized Experience Replay**:
   - Sample transitions with high TD error more frequently
   - Focuses learning on surprising or difficult experiences
   - Requires tracking and updating priorities

4. **Noisy Networks**:
   - Add parametric noise to network weights
   - Provides state-dependent exploration (vs epsilon-greedy)
   - Automatically reduces noise as learning progresses
"""


# Cell 12 - Practical Implementation Tips
"""
## Practical Implementation Tips

1. **Learning Rate Scheduling**:
   - Start with a higher learning rate and decrease over time
   - Example: 
     ```python
     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
     ```

2. **Target Network Updates**:
   - Hard updates: Copy weights periodically
     ```python
     target_net.load_state_dict(policy_net.state_dict())
     ```
   - Soft updates: Blend weights continuously
     ```python
     for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
         target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
     ```

3. **Gradient Clipping**:
   - Prevents exploding gradients
   - Example:
     ```python
     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     ```

4. **Environment Preprocessing**:
   - Frame stacking: Combine multiple observations for temporal information
   - Reward scaling: Normalize rewards for stable learning
   - State normalization: Scale inputs to reasonable ranges
"""
