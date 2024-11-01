import numpy as np
import random
import unittest
from uav_sim.envs.uav_sim import UavSim
from uav_sim.agents.uav import Obstacle, UAV, ObsType, AgentType, Entity, Cylinder, Platform, Vertiport, Grid

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

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

# Check if vertiport is initialized correctly
print("\nInitialized Vertiport Cylinders:")
for cylinder in env.vertiport.platform.platforms + env.vertiport.connector.connectors:
    print(f"Cylinder: Position {cylinder.pos}, Radius {cylinder.r}, Height {cylinder.height}")


# Function to test collision detection with detailed internal state setting
def test_collision_with_internal_state(uav_position, obstacle_position, cylinder_position):
    # Set the positions to test collision
    env.uavs[0]._state[0:3] = uav_position  # Directly modify the internal state
    env.uavs[1]._state[0:3] = np.array([0.0, 0.0, 10.0])  # Place UAV 1 far away to avoid interference

    # Set obstacle position to test collision
    env.obstacles[0]._state[0:3] = obstacle_position  # Directly modify the internal state
    for obstacle in env.obstacles[1:]:
        obstacle._state[0:3] = np.array([10.0, 10.0, 10.0])  # Place other obstacles far away

    # Set vertiport cylinder position to test collision
    env.vertiport.platform.platforms[0]._state[0:3] = cylinder_position  # Directly modify the internal state
    for cylinder in env.vertiport.platform.platforms[1:] + env.vertiport.connector.connectors:
        cylinder._state[0:3] = np.array([10.0, 10.0, 10.0])  # Place other cylinders far away

    # Step with zero velocity to keep positions static
    actions = {uav_id: np.zeros(3) for uav_id in env.uavs}
    obs, reward, done, truncated, info = env.step(actions)

    # Check collision with other UAVs
    collision_with_uav = info[0]["uav_collision"]
    print(f"Collision with UAV detected: {collision_with_uav}")

    # Check collision with obstacles
    collision_with_obstacle = info[0].get("obstacle_collision", False)
    print(f"Collision with obstacle detected: {collision_with_obstacle}")

    # Check collision with vertiport cylinder
    collision_with_cylinder = info[0].get("vertiport_collision", False)
    print(f"Collision with vertiport cylinder detected: {collision_with_cylinder}")

    # Assertions for testing
    assert collision_with_uav == True, "Expected collision with another UAV."
    assert collision_with_obstacle == True, "Expected collision with an obstacle."
    assert collision_with_cylinder == True, "Expected collision with a vertiport cylinder."


# Test collision with another UAV
print("\nTesting collision with another UAV:")
test_collision_with_internal_state(uav_position=np.array([0.0, 0.0, 9.8]),
                                   obstacle_position=np.array([10.0, 10.0, 10.0]),
                                   cylinder_position=np.array([10.0, 10.0, 10.0]))

# Test collision with an obstacle
print("\nTesting collision with an obstacle:")
test_collision_with_internal_state(uav_position=np.array([0.0, 0.0, 0.5]),
                                   obstacle_position=np.array([0.0, 0.0, 0.5]),
                                   cylinder_position=np.array([10.0, 10.0, 10.0]))

# Test collision with a vertiport cylinder
print("\nTesting collision with a vertiport cylinder:")
test_collision_with_internal_state(uav_position=np.array([0.0, 0.0, 0.5]),
                                   obstacle_position=np.array([10.0, 10.0, 10.0]),
                                   cylinder_position=np.array([0.0, 0.0, 0.5]))

print("\nAll collision tests passed!")
