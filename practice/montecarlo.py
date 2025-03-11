import gymnasium as gym
import numpy as np
from tqdm import tqdm


# First-visit MC prediction (Policy Evaluation)
def first_visit_mc_prediction(env, policy, gamma=0.9, episodes=5000, max_steps=100):
    V = np.zeros(env.observation_space.n)
    returns = {s: [] for s in range(env.observation_space.n)}

    for _ in tqdm(range(episodes), desc="Episodes"):
        states, rewards = [], []
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            states.append(state)
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            steps += 1

        # Add a small penalty if we didn't reach a terminal state
        if not done and len(rewards) > 0:
            rewards[-1] -= 0.1

        G = 0
        visited_states = set()
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            if states[t] not in visited_states:
                returns[states[t]].append(G)
                V[states[t]] = np.mean(returns[states[t]])
                visited_states.add(states[t])

    return V


# Monte Carlo Exploring Starts (ES) for control (policy iteration)
def monte_carlo_es(env, gamma=0.9, episodes=5000, max_steps=100):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = {
        (s, a): []
        for s in range(env.observation_space.n)
        for a in range(env.action_space.n)
    }
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)

    for _ in tqdm(range(episodes), desc="Episodes"):
        state, _ = env.reset()
        action = np.random.choice(env.action_space.n)  # exploring start
        episode = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

            # Add some exploration during learning
            if np.random.random() < 0.1:  # 10% exploration
                action = np.random.choice(env.action_space.n)
            else:
                action = policy[state] if not done else None

            steps += 1

        # Add a small penalty if we didn't reach a terminal state
        if not done and len(episode) > 0:
            _, _, old_reward = episode[-1]
            episode[-1] = (episode[-1][0], episode[-1][1], old_reward - 0.1)

        G = 0
        visited_state_action_pairs = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited_state_action_pairs:
                returns[(state, action)].append(G)
                Q[state, action] = np.mean(returns[(state, action)])
                policy[state] = np.argmax(Q[state])
                visited_state_action_pairs.add((state, action))

    return policy, Q


# Testing in FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Define an arbitrary deterministic policy (always move right)
policy = np.zeros(env.observation_space.n, dtype=int)

# Run First-visit MC Prediction
V = first_visit_mc_prediction(env, policy)
print("State Value Function V:\n", V)

# Run Monte Carlo ES
optimal_policy, Q = monte_carlo_es(env)
print("\nOptimal Policy π:\n", optimal_policy)
print("\nState-Action Value Function Q:\n", Q)
