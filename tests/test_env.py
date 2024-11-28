import unittest
import numpy as np
from uav_sim.envs.uav_sim import UavSim
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TestUavSimEnvironment(unittest.TestCase):
    def setUp(self):
        # Default environment configuration for testing
        self.env_config = {
            "dt": 0.1,
            "seed": 124,
            "num_uavs": 2,
            "num_obstacles": 4,
            "obstacle_radius": 0.25,
            "uav_radius": 0.1,
            "collision_delta": 0.1,
            "max_time": 50.0,
            "max_num_obstacles": 4,
            "max_velocity": 0.5,
            "max_acceleration": 0.5,
            "destination_reached_reward": 500.0,
            "collision_penalty": -100.0,
            "out_of_bounds_penalty": -100.0,
            "time_step_penalty": -2.0,
            "towards_dest_reward": 1.0,
            "exceed_acc_penalty": -1.0,
            "exceed_vel_penalty": -1.0,
            "env_max_w": 5.0,
            "env_max_l": 5.0,
            "env_max_h": 5.0,
            "env_min_h": 0.0,
            "z_high": 4.0,
            "z_low": 1.0,
        }
        self.env = UavSim(self.env_config)

    def test_action_space(self):
        # Test if the action space is defined correctly
        print("______________________________________________________________")
        print("\nTesting Action Space...")
        action_space = self.env._get_action_space()
        print(f"Action Space: {action_space}")
        self.assertEqual(action_space.shape, (3,))
        self.assertTrue(np.all(action_space.low == -1))
        self.assertTrue(np.all(action_space.high == 1))

    def test_obstacle_initialization(self):
        # Test if obstacles are initialized correctly
        print("______________________________________________________________")
        print("\nTesting Obstacle Initialization...")
        self.assertEqual(len(self.env.obstacles), self.env.num_obstacles)
        for obstacle in self.env.obstacles:
            print(f"Obstacle Position: x={obstacle.x}, y={obstacle.y}, z={obstacle.z}")
            self.assertTrue(-self.env.env_max_w <= obstacle.x <= self.env.env_max_w)
            self.assertTrue(-self.env.env_max_l <= obstacle.y <= self.env.env_max_l)
            self.assertTrue(self.env._z_low <= obstacle.z <= self.env._z_high)

    def test_observation_space(self):
        # Test if the observation space is defined correctly
        print("______________________________________________________________")
        print("\nTesting Observation Space...")
        obs_space = self.env._get_observation_space()
        print(f"Observation Space: {obs_space}")
        self.assertIn("self_state", obs_space.spaces)
        self.assertIn("other_uav_obs", obs_space.spaces)
        self.assertIn("obstacles", obs_space.spaces)
        self.assertIn("relative_destination", obs_space.spaces)

    def test_step_function(self):
        # Test if the step function updates UAV positions correctly
        print("______________________________________________________________")
        print("\nTesting Step Function...")
        # Reset the environment before testing
        self.env.reset()
        actions = {uav_id: np.array([0.1, 0.1, 0.1]) for uav_id in self.env.agent_ids}
        obs, reward, done, truncated, info = self.env.step(actions)
        print(f"Observations: {obs}\nRewards: {reward}\nDone: {done}\nTruncated: {truncated}\nInfo: {info}")
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, dict)
        self.assertIsInstance(done, dict)
        self.assertIsInstance(truncated, dict)
        self.assertIsInstance(info, dict)

    def test_reset_function(self):
        # Test if the environment resets correctly
        print("______________________________________________________________")
        print("\nTesting Reset Function...")
        obs, info = self.env.reset()
        print(f"Observations after reset: {obs}\nInfo after reset: {info}")
        self.assertEqual(len(self.env.uavs), self.env.num_uavs)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

    def test_two_steps(self):
        # Test taking two sequential steps in the environment
        print("______________________________________________________________")
        print("\nTesting Two Sequential Steps...")
        # Reset the environment before testing
        self.env.reset()
        actions_step_1 = {uav_id: np.array([1.0, 1.0, 1.0]) for uav_id in self.env.agent_ids}
        obs_1, reward_1, done_1, truncated_1, info_1 = self.env.step(actions_step_1)
        print(f"Step 1 - Observations: {obs_1}\nRewards: {reward_1}\nDone: {done_1}\nTruncated: {truncated_1}\nInfo: {info_1}")

        actions_step_2 = {uav_id: np.array([-1.0, -1.0, -1.0]) for uav_id in self.env.agent_ids}
        obs_2, reward_2, done_2, truncated_2, info_2 = self.env.step(actions_step_2)
        print(f"Step 2 - Observations: {obs_2}\nRewards: {reward_2}\nDone: {done_2}\nTruncated: {truncated_2}\nInfo: {info_2}")

        self.assertIsInstance(obs_2, dict)
        self.assertIsInstance(reward_2, dict)
        self.assertIsInstance(done_2, dict)
        self.assertIsInstance(truncated_2, dict)
        self.assertIsInstance(info_2, dict)

    def test_step_out_of_bounds_function(self):
        # Test if the step function updates UAV positions correctly and detects out-of-bounds condition
        print("______________________________________________________________")
        print("\nTesting Step Out of Bounds Function...")
        # Reset the environment before testing
        self.env.reset()
        # Assigning large actions to simulate going out of bounds
        actions2 = {uav_id: np.array([10.0, 10.0, 10.0]) for uav_id in self.env.agent_ids}
        for i in range(0, 20):
            self.env.step(actions2)
        # Taking a step in the environment
        obs, reward, done, truncated, info = self.env.step(actions2)
        # Printing out the details for debugging
        print(f"Observations: {obs}\nRewards: {reward}\nDone: {done}\nTruncated: {truncated}\nInfo: {info}")

        # Assertions to verify the data types
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, dict)
        self.assertIsInstance(done, dict)
        self.assertIsInstance(truncated, dict)
        self.assertIsInstance(info, dict)


    def test_single_uav_reached_target_and_collision(self):
        # Test if a single UAV reaches the target and if there is a collision
        print("______________________________________________________________")
        print("\nTesting Single UAV Reached Target and Collision...")
        # Reset the environment before testing
        self.env.reset()
        uav = list(self.env.uavs.values())[0]

        # Set UAV destination close to the current position to ensure reaching the target
        uav.set_new_destination(np.array([uav.x, uav.y, uav.z]))
        reached, _, _ = uav.check_dest_reached()  # Update to use the `check_dest_reached` method correctly
        print(f"UAV reached target: {reached}")
        self.assertTrue(reached)

        # Manually create a collision scenario
        other_entity = self.env.obstacles[0]
        other_entity.update_state(uav.pos, np.zeros(3))  # Place the obstacle at the UAV's position
        in_collision = uav.in_collision(other_entity)
        print(f"UAV collided with obstacle: {in_collision}")
        self.assertTrue(in_collision)

    def test_two_uav_collision(self):
        # Test if two UAVs reach the target and if there is a collision between them
        print("______________________________________________________________")
        print("\nTesting Two UAVs Collision...")
        # Reset the environment before testing
        self.env.reset()
        uav_1, uav_2 = list(self.env.uavs.values())

        # Create a collision scenario between both UAVs
        uav_2.update_state(uav_1.pos, np.zeros(3))  # Place UAV 2 at UAV 1's position

        # Actions to move UAVs toward their respective destinations (small step)
        actions = {
            uav_1.id: np.array([0.0, 0.0, 0.0]),  # UAV 1 remains stationary (already at destination)
            uav_2.id: np.array([0.0, 0.0, 0.0]),  # UAV 2 remains stationary (already at destination)
        }

        # Take a step in the environment
        obs, reward, done, truncated, info = self.env.step(actions)

        in_collision = uav_1.in_collision(uav_2)
        print(f"Collision between UAV 1 and UAV 2: {in_collision}")
        self.assertTrue(in_collision)

        # Print reward and info
        print(f"Rewards: {reward}")
        print(f"Info: {info}")

        # Assertions for reward and info types
        self.assertIsInstance(reward, dict)
        self.assertIsInstance(info, dict)

    def test_two_uav_reached_target(self):
        # Test if two UAVs reach the target and if there is a collision between them
        print("______________________________________________________________")
        print("\nTesting Two UAVs Reached Target...")
        # Reset the environment before testing
        self.env.reset()
        uav_1, uav_2 = list(self.env.uavs.values())

        # Set both UAVs to have the same target position to check if they both reach it
        destination1 = np.array([uav_1.x, uav_1.y, uav_1.z])
        destination2 = np.array([uav_2.x, uav_2.y, uav_2.z])
        uav_1.set_new_destination(destination1)
        uav_2.set_new_destination(destination2)

        # Actions to move UAVs toward their respective destinations (small step)
        actions = {
            uav_1.id: np.array([0.0, 0.0, 0.0]),  # UAV 1 remains stationary (already at destination)
            uav_2.id: np.array([0.0, 0.0, 0.0]),  # UAV 2 remains stationary (already at destination)
        }

        # Take a step in the environment
        obs, reward, done, truncated, info = self.env.step(actions)

        # Check if UAVs reach the destination
        reached_1, _, _ = uav_1.check_dest_reached()
        reached_2, _, _ = uav_2.check_dest_reached()
        print(f"UAV 1 reached target: {reached_1}")
        print(f"UAV 2 reached target: {reached_2}")

        self.assertTrue(reached_1)
        self.assertTrue(reached_2)

        # Print reward and info
        print(f"Rewards: {reward}")
        print(f"Info: {info}")

        # Assertions for reward and info types
        self.assertIsInstance(reward, dict)
        self.assertIsInstance(info, dict)
    def test_clip_velocity(self):
        # Test if velocity is clipped correctly
        print("______________________________________________________________")
        print("\nTesting Clip Velocity...")
        velocity = np.array([0.6, 0.6, 0.6])
        clipped_velocity = self.env._clip_velocity(velocity)
        print(f"Original Velocity: {velocity}\nClipped Velocity: {clipped_velocity}")
        self.assertTrue(np.linalg.norm(clipped_velocity) <= self.env.max_velocity)

    def test_get_info(self):
        # Test if UAV information is collected correctly
        print("______________________________________________________________")
        print("\nTesting Get Info...")
        # Reset the environment before testing
        self.env.reset()
        uav = list(self.env.uavs.values())[0]
        info = self.env._get_info(uav)
        print(f"UAV Info: {info}")
        self.assertIn("time_step", info)
        self.assertIn("uav_collision", info)
        self.assertIn("obstacle_collision", info)
        self.assertIn("uav_rel_dist", info)
        self.assertIn("uav_rel_vel", info)
        self.assertIn("uav_reached_dest", info)
        self.assertIn("uav_out_of_bounds", info)

    def test_get_obs(self):
        # Test if observations are generated correctly for a UAV
        print("______________________________________________________________")
        print("\nTesting Get Observations...")
        # Reset the environment before testing
        self.env.reset()
        uav = list(self.env.uavs.values())[0]
        obs = self.env._get_obs(uav)
        print(f"UAV Observations: {obs}")
        self.assertIn("self_state", obs)
        self.assertIn("other_uav_obs", obs)
        self.assertIn("obstacles", obs)
        self.assertIn("relative_destination", obs)

    def test_get_reward(self):
        # Test if rewards are calculated correctly
        print("______________________________________________________________")
        print("\nTesting Get Reward...")
        # Reset the environment before testing
        self.env.reset()
        uav = list(self.env.uavs.values())[0]
        reward = self.env._get_reward(uav)
        print(f"Reward: {reward}")
        self.assertIsInstance(reward, float)

    def test_is_out_of_bounds(self):
        # Test if UAV is correctly identified as out of bounds
        print("______________________________________________________________")
        print("\nTesting Out of Bounds Check...")
        # Reset the environment before testing
        self.env.reset()
        uav = list(self.env.uavs.values())[0]
        uav.update_state(np.array([10.0, 10.0, 10.0]), np.zeros(3))
        out_of_bounds = self.env.is_out_of_bounds(uav)
        print(f"UAV Position: {uav.pos}, Out of Bounds: {out_of_bounds}")
        self.assertTrue(out_of_bounds)

    def test_initialize_uavs(self):
        # Test if UAVs are initialized correctly during reset
        print("______________________________________________________________")
        print("\nTesting UAV Initialization...")
        self.env.reset()
        self.assertEqual(len(self.env.uavs), self.env.num_uavs)
        for uav in self.env.uavs.values():
            print(f"UAV Position: x={uav.x}, y={uav.y}, z={uav.z}")
            self.assertTrue(-self.env.env_max_w <= uav.x <= self.env.env_max_w)
            self.assertTrue(-self.env.env_max_l <= uav.y <= self.env.env_max_l)
            self.assertTrue(self.env.env_min_h <= uav.z <= self.env.env_max_h)


if __name__ == '__main__':
    unittest.main()
