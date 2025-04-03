# gym_utils.py

import gymnasium as gym
import numpy as np
import math
from typing import Optional # Optional is used for type hinting

def get_env_info(env: gym.Env) -> str:
    """
    Generates a formatted string containing information about a Gymnasium environment.

    Args:
        env: An initialized Gymnasium environment instance.

    Returns:
        A string summarizing the environment's spaces and attempting to render it.
    """
    info_lines = []
    # Use spec.id if available for the official registered name
    env_id = env.spec.id if env.spec else "Unknown Environment"
    info_lines.append(f"--- Environment Info: {env_id} ---")

    # --- Process Observation Space ---
    obs_space = env.observation_space
    info_lines.append(f"\nObservation Space: {obs_space}")
    info_lines.extend(get_space_details(obs_space, env_id, is_observation=True))

    # --- Process Action Space ---
    action_space = env.action_space
    info_lines.append(f"\nAction Space: {action_space}")
    info_lines.extend(get_space_details(action_space, env_id, is_observation=False))

    # --- Rendering ---
    # Check if render_mode is set and suitable (not 'human' which opens windows)
    render_mode = getattr(env, 'render_mode', None)
    render_modes_for_info = ['ansi', 'rgb_array']
    can_render_info = render_mode in render_modes_for_info

    if can_render_info:
        info_lines.append("\nEnvironment Layout/Render:")
        try:
            # Resetting might be needed for some envs to render correctly initially
            # Use try-except in case reset is problematic or not needed
            try:
                env.reset()
            except Exception:
                pass # Continue even if reset fails

            render_output = env.render()

            if render_mode == 'ansi':
                info_lines.append(render_output)
                # Add specific legends if known
                if env_id.startswith("FrozenLake"):
                    info_lines.append("\nLegend (FrozenLake): S=Start, F=Frozen, H=Hole, G=Goal, Highlight=Agent")
            elif render_mode == 'rgb_array':
                 info_lines.append(f"  (Rendered as RGB array with shape: {render_output.shape})")

        except Exception as e:
            info_lines.append(f"  (Could not render environment: {e})")
    else:
        info_lines.append(f"\nEnvironment Rendering not available/suitable for info string.")
        info_lines.append(f"  (Current render_mode: {render_mode}. Use 'ansi' or 'rgb_array' in gym.make for render info).")

    return "\n".join(info_lines)


def get_space_details(space: gym.Space, env_id: str, is_observation: bool) -> list[str]:
    """Helper function to get details for different space types."""
    details = []
    space_type_name = type(space).__name__
    details.append(f"  - Type: {space_type_name}")

    if isinstance(space, gym.spaces.Discrete):
        details.append(f"  - Size: {space.n}")
        if is_observation:
            # Try to guess grid shape for common grid worlds based on size
            if space.n > 1:
                sqrt_n = math.isqrt(space.n)
                if sqrt_n * sqrt_n == space.n:
                    grid_shape = (sqrt_n, sqrt_n)
                    details.append(f"  - Possible Grid Shape: {grid_shape}")
                    try:
                         # Attempt to show grid indexing
                         grid_indices = np.arange(space.n).reshape(grid_shape)
                         details.append("  - Typical State Indexing (if grid):")
                         # Format numpy array nicely
                         details.append(np.array2string(grid_indices, prefix="    "))
                    except ValueError:
                         details.append("  - (Could not format state indices as grid)")

        else: # Action space specific details
             details.append(f"  - Number of actions: {space.n}")
             # Add known mappings as examples - cannot be fully generalized
             if env_id.startswith("FrozenLake"):
                 details.append("  - Action Mapping (FrozenLake): {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}")
             elif env_id.startswith("CartPole"):
                 details.append("  - Action Mapping (CartPole): {0: 'Push left', 1: 'Push right'}")
             elif env_id.startswith("Taxi"):
                 details.append("  - Action Mapping (Taxi): {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}")
             # Add more known mappings here if desired
             else:
                 details.append("  - Action meanings specific to this environment.")

    elif isinstance(space, gym.spaces.Box):
        details.append(f"  - Shape: {space.shape}")
        details.append(f"  - Data Type: {space.dtype}")
        # Show bounds concisely, especially if uniform
        if np.all(space.low == space.low.item(0)) and np.all(space.high == space.high.item(0)):
             details.append(f"  - Low Bounds (uniform): {space.low.item(0)}")
             details.append(f"  - High Bounds (uniform): {space.high.item(0)}")
        else:
             details.append(f"  - Low Bounds: {space.low}")
             details.append(f"  - High Bounds: {space.high}")

    elif isinstance(space, gym.spaces.Tuple):
        details.append("  - Components:")
        for i, sub_space in enumerate(space.spaces):
             details.append(f"    Component {i}: {sub_space}")
             # Recursively get details, indented
             sub_details = get_space_details(sub_space, env_id, is_observation)
             details.extend([f"    {line}" for line in sub_details])

    elif isinstance(space, gym.spaces.Dict):
        details.append("  - Components:")
        for key, sub_space in space.spaces.items():
             details.append(f"    Component '{key}': {sub_space}")
             # Recursively get details, indented
             sub_details = get_space_details(sub_space, env_id, is_observation)
             details.extend([f"    {line}" for line in sub_details])

    # Add elif blocks here for other space types like MultiDiscrete, MultiBinary as needed

    return details


# --- Example Usage (if you run this file directly) ---
if __name__ == "__main__":
    print("Demonstrating get_env_info function...\n")

    # List of environments to test
    env_names_to_test = [
        ("FrozenLake-v1", {'is_slippery': False, 'render_mode': 'ansi'}),
        ("CartPole-v1", {'render_mode': 'rgb_array'}),
        ("Pendulum-v1", {'render_mode': 'rgb_array'}),
        ("Taxi-v3", {'render_mode': 'ansi'}),
        # ("BipedalWalker-v3", {'render_mode': 'rgb_array'}), # Example with Box actions
        # ("Blackjack-v1", {}), # Example with Tuple observation space
    ]

    for name, kwargs in env_names_to_test:
        print("\n" + "="*20 + f" {name} Example " + "="*20)
        env: Optional[gym.Env] = None # Initialize env to None
        try:
            # Use context manager for clean handling
            with gym.make(name, **kwargs) as env:
                info_str = get_env_info(env)
                print(info_str)
        except ImportError as e:
             print(f"Could not run example for {name}. Dependencies might be missing: {e}")
        except Exception as e:
            print(f"Could not create or get info for {name}: {e}")
            if env: # Close env if creation succeeded but info failed
                 try:
                     env.close()
                 except Exception:
                     pass # Ignore errors during cleanup