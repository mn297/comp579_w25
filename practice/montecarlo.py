import gymnasium as gym
import numpy as np
from tqdm import tqdm


# First-visit MC prediction (Policy Evaluation)
def first_visit_mc_prediction(env, policy, gamma=0.9, episodes=5000):
    V = np.zeros(env.observation_space.n)
    returns = {s: [] for s in range(env.observation_space.n)}

    for _ in tqdm(range(episodes), desc="Episodes"):
        states, rewards = [], []
        state, _ = env.reset()
        done = False

        while not done:
            states.append(state)
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)

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
def monte_carlo_es(env, gamma=0.9, episodes=5000):
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

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            action = policy[state] if not done else None

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
