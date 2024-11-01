import numpy as np
import random
import unittest
from uav_sim.envs.uav_sim import UavSim
from uav_sim.agents.uav import Obstacle, UAV, ObsType, AgentType, Entity, Cylinder, Platform, Vertiport, Grid


# Set a random seed for reproducibility
np.random.seed(43)
random.seed(43)

# Define a sample configuration for the environment
env_config = {
    "dt": 0.1,
    "seed": 123,
    "num_uavs": 2,
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

# Test for Collision Event
print("\nTesting Collision Event:")
# Set the position of the UAVs to be very close to each other or near an obstacle to simulate a collision
env.uavs[0]._state[0:3] = np.array([0.0, 0.0, 0.5])
env.uavs[1]._state[0:3] = np.array([0.0, 0.0, 0.6])  # Very close to UAV 0, should cause a collision

actions = {uav_id: np.zeros(3) for uav_id in env.uavs}  # No movement actions

obs, reward, done, truncated, info = env.step(actions)

# Check for collision
collision_detected = any(info[uav_id]["uav_collision"] for uav_id in env.uavs)
print("Collision Detected:", collision_detected)
assert collision_detected, "Collision was expected but not detected."

# Test for Reaching Destination
print("\nTesting Reaching Destination Event:")
# Set the position of the UAV close to its destination
for uav_id, uav in env.uavs.items():
    uav.destination = np.array([0.0, 0.0, 0.5])
    uav._state[0:3] = np.array([0.0, 0.0, 0.49])  # Very close to destination

actions = {uav_id: np.zeros(3) for uav_id in env.uavs}  # No movement actions

obs, reward, done, truncated, info = env.step(actions)

# Check if the UAV reached its destination
for uav_id, uav_info in info.items():
    reached_dest = uav_info["uav_reached_dest"]
    print(f"UAV {uav_id} reached destination:", reached_dest)
    assert reached_dest, f"UAV {uav_id} was expected to reach destination but didn't."

# Test for Out of Bounds Event
print("\nTesting Out of Bounds Event:")
# Set the position of the UAV out of bounds
env.uavs[0]._state[0:3] = np.array([6.0, 0.0, 0.5])  # Out of bounds on x-axis
env.uavs[1]._state[0:3] = np.array([0.0, 6.0, 0.5])  # Out of bounds on y-axis

actions = {uav_id: np.zeros(3) for uav_id in env.uavs}  # No movement actions

obs, reward, done, truncated, info = env.step(actions)

# Check for out of bounds
for uav_id, uav_info in info.items():
    out_of_bounds = uav_info["uav_out_of_bounds"]
    print(f"UAV {uav_id} out of bounds:", out_of_bounds)
    assert out_of_bounds, f"UAV {uav_id} was expected to be out of bounds but wasn't."

print("\nAll event tests passed!")

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
