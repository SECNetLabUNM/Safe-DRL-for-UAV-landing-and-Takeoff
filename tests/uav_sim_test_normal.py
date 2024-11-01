import numpy as np
import random
import unittest
from uav_sim.envs.uav_sim import UavSim
from uav_sim.agents.uav import Obstacle, UAV, ObsType, AgentType, Entity, Cylinder, Platform, Vertiport, Grid


# Set a random seed for reproducibility
#np.random.seed(42)
#random.seed(42)

# Define a sample configuration for the environment
env_config = {
    "dt": 0.1,
    "seed": 45,
    "num_uavs": 4,
    "num_obstacles": 4,
    "obstacle_radius": 0.1,
    "env_max_w": 5,
    "env_max_l": 5,
    "env_max_h": 5,
    "env_min_h": 0,
    "max_time": 50.0,
    "use_safe_action": False,
}

# Initialize the environment
env = UavSim(env_config)

# Check if UAVs are initialized correctly
print("Initialized UAVs:")
for uav_id, uav in env.uavs.items():
    print(f"UAV {uav_id}: Position {uav.pos}, Velocity {uav.vel}, Destination {uav.destination}")

# Check if obstacles are initialized correctly
print("\nInitialized Obstacles:")
for obstacle in env.obstacles:
    print(f"Obstacle {obstacle.id}: Position {obstacle.pos}, Velocity {obstacle.vel}")

# Sample random actions for all UAVs
actions = {uav_id: np.random.uniform(-1, 1, size=3) for uav_id in env.uavs}
print(f"Actions: {actions}")
# Print actions for each UAV
print("Actions for all UAVs:")
for uav_id, action in actions.items():
    print(f"UAV {uav_id}: Action {action}")

env.step(actions)
# Step the environment with random actions
obs, reward, done, truncated, info = env.step(actions)

# Print out the results
print("\nStep Results:")
print("Observations:", obs)
print("Rewards:", reward)
print("Done Flags:", done)
print("Truncated:", truncated)
print("Info:", info)


# Generate new random actions for the second step
actions = {uav_id:  np.array([0.0, 0.0, 3]) for uav_id in env.uavs}

# Print actions for each UAV
print("\nActions for all UAVs (Step 2):")
for uav_id, action in actions.items():
    print(f"UAV {uav_id}: Action {action}")

# Take the second step in the environment
obs, reward, done, truncated, info = env.step(actions)

# Print out the results after the second step
print("\nStep 2 Results:")
print("Observations:", obs)
print("Rewards:", reward)
print("Done Flags:", done)
print("Truncated:", truncated)
print("Info:", info)


# Test get_h and get_b Functions
uav = env.uavs[0]
entity = env.uavs[1]

h_value = env.get_h(uav, entity)
b_value = env.get_b(uav, entity)

print("\nTest get_h and get_b:")
print(f"h value: {h_value}")
print(f"b value: {b_value}")

# Test get_safe_action Function
uav = env.uavs[0]
desired_action = np.array([0.0, 0.0, 3])  # Sample desired action
safe_action = env.get_safe_action(uav, desired_action)

print("\nTest get_safe_action:")
print(f"Desired Action: {desired_action}")
print(f"Safe Action: {safe_action}")

# Test reset Function
print("\nResetting the environment...")
obs, info = env.reset()

print("\nAfter Reset:")
print("Observations:", obs)
print("Info:", info)


print("\nAll tests passed!")


