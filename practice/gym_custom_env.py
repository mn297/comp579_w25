import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleGridWorldEnv(gym.Env):
    """
    A simple 5x5 grid world where the agent must reach the goal.
    """

    def __init__(self):
        super(SimpleGridWorldEnv, self).__init__()

        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Define observation space: agent position (x,y)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.float32)

        # Set goal position
        self.goal = np.array([4, 4])

    def reset(self, seed=None, options=None):
        # Reset agent to starting position
        self.agent_pos = np.array([0, 0])

        # New in Gymnasium: need to pass the seed to the super class
        super().reset(seed=seed)

        # Return initial observation and empty info dict
        return self.agent_pos.copy(), {}

    def step(self, action):
        # Move agent based on action
        if action == 0:  # up
            self.agent_pos[1] = min(self.agent_pos[1] + 1, 4)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, 4)
        elif action == 2:  # down
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)

        # Calculate reward (-0.1 per step, +10 for reaching goal)
        reward = -0.1

        # Check if agent reached goal
        done = np.array_equal(self.agent_pos, self.goal)
        if done:
            reward = 10.0

        # Return observation, reward, done, truncated, info
        return self.agent_pos.copy(), reward, done, False, {}

    def render(self):
        # Simple text-based rendering
        grid = np.zeros((5, 5), dtype=str)
        grid.fill(".")
        grid[self.goal[1], self.goal[0]] = "G"
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"

        for row in reversed(grid):
            print(" ".join(row))
        print("\n")


# Test the environment
env = SimpleGridWorldEnv()
state, _ = env.reset()
print("Initial state:", state)
env.render()

# Take some random actions
for i in range(10):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, New state: {state}, Reward: {reward}")
    env.render()

    if done:
        print("Goal reached!")
        break
