import sys
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from uav_sim.agents.uav import Obstacle, UAV, ObsType, AgentType, Entity, Grid
import logging
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UavSim(MultiAgentEnv):

    def __init__(self, env_config={}):
        super().__init__()
        self.dt = env_config.setdefault("dt", 0.1)
        self._seed = env_config.setdefault("seed", None)
        self.seed(self._seed)
        self.num_uavs = env_config.setdefault("num_uavs", 2)
        self.num_obstacles = env_config.setdefault("num_obstacles", 4)
        self.obstacle_radius = env_config.setdefault("obstacle_radius", 0.25)
        self.uav_radius = env_config.setdefault("uav_radius", 0.25)
        self.collision_delta = env_config.setdefault("collision_delta", 0.1)  # Set collision delta

        self.max_time = env_config.setdefault("max_time", 50.0)
        self.max_num_obstacles = env_config.setdefault("max_num_obstacles", 4)

        if self.max_num_obstacles < self.num_obstacles:
            self.num_obstacles = self.max_num_obstacles

        self.max_velocity = env_config.setdefault("max_velocity", 0.5)  # Set max velocity
        self.max_acceleration = env_config.setdefault("max_acceleration", 0.5)  # Set max acceleration

        self.reward_params = {
            "destination_reached_reward": env_config.get("destination_reached_reward", 500.0),
            "collision_penalty": env_config.get("collision_penalty", -100.0),
            "out_of_bounds_penalty": env_config.get("out_of_bounds_penalty", -100.0),
            "time_step_penalty": env_config.get("time_step_penalty", -2.0),
            "towards_dest_reward": env_config.get("towards_dest_reward", 1.0),
            "exceed_acc_penalty": env_config.get("exceed_acc_penalty", -1.0),
            "exceed_vel_penalty": env_config.get("exceed_vel_penalty", -1.0)
        }

        self._agent_ids = set(range(self.num_uavs))
        self._uav_type = getattr(
            sys.modules[__name__], env_config.get("uav_type", "UAV")
        )

        self.env_max_w = env_config.setdefault("env_max_w", 5.0)
        self.env_max_l = env_config.setdefault("env_max_l", 5.0)
        self.env_max_h = env_config.setdefault("env_max_h", 5.0)
        self.env_min_h = env_config.setdefault("env_min_h", 0.0)

        self._z_high = env_config.setdefault("z_high", 4.0)
        self._z_high = min(self.env_max_h, self._z_high)
        self._z_low = env_config.setdefault("z_low", 1.0)
        self._z_low = max(0, self._z_low)

        # Create a Grid instance
        self.grid_1 = Grid(grid_size=4, spacing=1.0, z_height=0.5)
        self.grid_2 = Grid(grid_size=4, spacing=1.0, z_height=4.5)

        self._env_config = env_config
        self.norm_action_high = np.ones(3)
        self.norm_action_low = np.ones(3) * -1

        self.action_high = np.ones(3) * 1.0
        self.action_low = np.ones(3) * -1

        self._time_elapsed = 0.0

        # Initialize obstacles and UAVs
        self.obstacles = self._initialize_obstacles()
        self.uavs = {}

        self.reset()
        #print(self.uavs)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    @property
    def env_config(self):
        return self._env_config

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @property
    def agent_ids(self):
        return self._agent_ids

    def _get_action_space(self):
        """The action of the UAV. It consists of acceleration in x, y, and z components."""
        return spaces.Box(
                    low=np.float32(self.action_low),
                    high=np.float32(self.action_high),
                    shape=(3,),
                    dtype=np.float32,
                )

    def _initialize_obstacles(self):
        """Initialize obstacles with random positions."""
        obstacles = []
        for i in range(self.num_obstacles):
            obstacle = Obstacle(
                _id=i,
                x=self.np_random.uniform(-self.env_max_w, self.env_max_w),
                y=self.np_random.uniform(-self.env_max_l, self.env_max_l),
                z=self.np_random.uniform(self._z_low, self._z_high),
                r=self.obstacle_radius,
                collision_delta=self.collision_delta,  # Apply collision delta to obstacles
                _type=ObsType.M,
                # _type=ObsType.M if i % 2 == 0 else ObsType.S,
            )
            obstacles.append(obstacle)
        return obstacles

    def _initialize_uavs(self, num_uav_grid_1_start, num_uav_grid_2_start, grid_1_points, grid_2_points):
        """Initialize UAVs, ensuring no collisions at start"""
        grid_1_points_copy = grid_1_points.copy()
        #(f"grid_1_points_copy: {grid_1_points_copy}")
        grid_2_points_copy = grid_2_points.copy()
        #print(f"grid_2_points_copy: {grid_2_points_copy}")
        # Assign UAVs that start on the grid and move to platform points
        for agent_id in range(num_uav_grid_1_start):
            # Get a random grid point as the start position
            start_position = grid_1_points.pop()
            # Get a platform point as the destination
            destination = grid_2_points.pop()

            # Initialize the UAV with the grid start position and platform as the destination
            uav = self._uav_type(
                _id=agent_id,
                x=start_position[0],
                y=start_position[1],
                z=start_position[2],
                r=self.uav_radius,
                dt=self.dt,
                collision_delta=self.collision_delta,
            )

            # Set the UAV's destination after initialization
            uav.set_new_destination(destination=destination)  # Set destination as needed

            # Set initial distance to destination
            uav.last_rel_dist = np.linalg.norm([uav.x, uav.y, uav.z] - np.array(destination))
            self.uavs[agent_id] = uav

        # Assign UAVs that start at the platform and move to grid points
        for agent_id in range(num_uav_grid_1_start, self.num_uavs):
            # Get a random platform point as the start position
            start_position = grid_2_points_copy.pop()
            # Get a random grid point as the destination
            destination = grid_1_points_copy.pop()
            # Initialize the UAV with platform start position and grid point as the destination
            uav = self._uav_type(
                _id=agent_id,
                x=start_position[0],
                y=start_position[1],
                z=start_position[2],
                r=self.uav_radius,
                dt=self.dt,
                collision_delta=self.collision_delta,
            )
            # Set the UAV's destination after initialization
            uav.set_new_destination(destination=destination)  # Set destination as needed

            # Set initial distance to destination
            uav.last_rel_dist = np.linalg.norm([uav.x, uav.y, uav.z] - np.array(destination))
            self.uavs[agent_id] = uav

    def _get_observation_space(self):
        entity_state_shape = 11  # pos (3), vel (3), radius (1), direction (3), height (1)

        self_state_shape = entity_state_shape
        num_other_uavs = self.num_uavs - 1
        other_uav_shape = (num_other_uavs, entity_state_shape)
        #print(f"other_uav_shape: {other_uav_shape}")
        obstacle_shape = (self.num_obstacles, entity_state_shape)
        destination_shape = 3  # Destination's position only
        #print(f"num_other_uavs: {num_other_uavs}")

        obs_space = spaces.Dict(
            {
                "self_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self_state_shape,),
                    dtype=np.float32,
                ),
                "other_uav_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=other_uav_shape,
                    dtype=np.float32,
                ),
                "obstacles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obstacle_shape,
                    dtype=np.float32,
                ),
                "relative_destination": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(destination_shape,),
                    dtype=np.float32,
                ),
            }
        )
        # Print the overall observation shapes
        #for key, value in obs_space.items():
            #print(f"{key} shape: {value.shape}")

        return obs_space

    def _clip_velocity(self, velocity):
        """
        Clamps the velocity vector to ensure that its magnitude does not exceed the maximum allowed velocity.

        :param velocity: The velocity vector [vx, vy, vz].
        :return: A velocity vector that respects the maximum velocity limit.
        """
        max_velocity = self.max_velocity  # Example: max velocity set to 5

        # Calculate the magnitude of the velocity vector
        magnitude = np.linalg.norm(velocity)
        #print(f"magnitude velocity:{magnitude}")
        # If the magnitude exceeds the maximum velocity, scale the velocity
        if magnitude > max_velocity:
            # Normalize and scale the velocity to the maximum allowable magnitude
            velocity = (velocity / magnitude) * max_velocity

        return velocity

    def step(self, actions):
        """
        Update the environment for each UAV based on the actions taken, ensuring collision avoidance
        with spheres (UAVs, moving obstacles) and cylinders (vertiport components).

        :param actions: A dictionary where the keys are UAV IDs and the values are the actions for each UAV.
        :return: Updated observations, rewards, done flags, truncated flags, and additional info.
        """
        # Track which agents are still active (i.e., haven't finished)
        self.alive_agents = set()
        #print(f"Actions:{actions}")
        for i, action in actions.items():
            self.alive_agents.add(i)

            # Skip if the UAV is already done
            if self.uavs[i].done:
                continue
        #print(f"action:{action}")
        #print(f"action type :{type(action)}")

        # Ensure action (velocity) is within valid bounds by applying velocity clipping
        # Make sure `action` is a numpy array or list before passing to `_clip_velocity`
        #if isinstance(action, dict):
            #print(f"Agent {i} action before clip: {action}")
       # else:
            #action = np.array(action)  # Convert to numpy array if necessary


            # Ensure action (velocity) is within valid bounds by applying velocity clipping
            action = self._clip_velocity(action)

            # Update the UAV's state using the given (or adjusted) action
            self.uavs[i].step(action)

        # Step the obstacles
        for obstacle in self.obstacles:
            # Moving obstacles update their positions, fixed ones do nothing
            obstacle.step()

        # Gather observations, rewards, done flags, and additional info
        obs = {uav.id: self._get_obs(uav) for uav in self.uavs.values() if uav.id in self.alive_agents}
        reward = {uav.id: self._get_reward(uav) for uav in self.uavs.values() if uav.id in self.alive_agents}

        # Collect additional information
        info = {
            uav.id: self._get_info(uav)
            for uav in self.uavs.values()
            if uav.id in self.alive_agents
        }


        # Calculate done flags for each UAV
        done = {
            self.uavs[uav_id].id: self.uavs[uav_id].done
            for uav_id in self.alive_agents
        }

        # Calculate truncated flags for each UAV
        truncated = {
            self.uavs[uav_id].id: self.time_elapsed >= self.max_time
            for uav_id in self.alive_agents
        }

        # Check if the simulation should be terminated or truncated
        # Determine if all UAVs have finished (landed or marked as done)
        done["__all__"] = all(uav.done for uav in self.uavs.values())
        truncated["__all__"] = self.time_elapsed >= self.max_time

        # Increment time
        self._time_elapsed += self.dt

        # Return observations, rewards, done flags, truncated flags, and additional info
        return obs, reward, done, truncated, info

    def _get_info(self, uav):
        """
        Collect information about the UAV at the current time step, including:
        - Time step
        - Collision with other UAVs
        - Collision with moving obstacles
        - Collision with vertiport components
        - Relative velocity and distance to destination
        - Whether the UAV has reached its destination
        - Whether the UAV is out of the environment range
        """
        # Check if UAV has reached its destination
        is_reached, rel_dist, rel_vel = uav.check_dest_reached()

        # Check for collisions with other UAVs
        uav_collision = any([uav.in_collision(other_uav) for other_uav in self.uavs.values() if uav.id != other_uav.id])

        # Check for collisions with moving obstacles
        obstacle_collision = any(
            [uav.in_collision(obstacle) for obstacle in self.obstacles if obstacle.type == ObsType.M])

        # Check if UAV is out of the environment bounds
        out_of_bounds = self.is_out_of_bounds(uav)

        # Collect the information
        info = {
            "time_step": self.time_elapsed,  # Current time step
            "uav_collision": 1.0 if uav_collision else 0.0,  # Collision with other UAVs
            "obstacle_collision": 1.0 if obstacle_collision else 0.0,  # Collision with moving obstacles
            "uav_rel_dist": rel_dist,  # Distance to destination
            "uav_rel_vel": rel_vel,  # Relative velocity to destination
            "uav_reached_dest": 1.0 if is_reached else 0.0,  # 1 if UAV has reached its destination, 0 otherwise
            "uav_out_of_bounds": 1.0 if out_of_bounds else 0.0  # 1 if UAV is out of bounds, 0 otherwise
        }

        return info

    def _get_obs(self, uav):
        """
        Returns the observation for a specific UAV, including:
        - Self state (position, velocity, radius, direction, height)
        - Other UAVs' states (position, velocity, radius, direction, height)
        - Moving obstacles' states (position, velocity, radius, direction, height)
        - Relative landing platform position (a point in the destination_info)
        - Vertiport components (position, velocity, radius, direction, height of all cylinders)
        """
        # Get self state (position, velocity, radius, direction, height)
        # print(f"UAV {uav.id} direction before concat: {uav.direction}")  # Debugging direction

        self_state = np.concatenate([
            uav.pos,  # Position (x, y, z)
            uav.vel,  # Velocity (x_dot, y_dot, z_dot)
            [uav.r],  # Radius
            uav.direction,  # Direction vector (dx, dy, dz)
            [uav.height]  # Height
        ]).astype(np.float32)

        # print(f"Self state shape: {self_state.shape}")  # Debugging print

        # Get other UAVs' states (position, velocity, radius, direction, height)
        other_uav_states = []
        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                state = np.concatenate([
                    other_uav.pos,
                    other_uav.vel,
                    [other_uav.r],
                    other_uav.direction,
                    [other_uav.height]
                ]).astype(np.float32)

                # print(f"Other UAV {other_uav.id} state shape: {state.shape}")  # Debugging print
                other_uav_states.append(state)

        # Ensure all other UAVs' observations are of the same shape
        other_uav_states = np.array(other_uav_states)

        # for obstacle in self.obstacles:
        # print(
        # f"Obstacle {obstacle.id}: Position {obstacle.pos}, Velocity {obstacle.vel}, Direction {obstacle.direction}, Height {obstacle.height}")

        # Get moving obstacles' states (position, velocity, radius, direction, height)
        obstacles_states = [
            np.concatenate([
                obstacle.pos,
                obstacle.vel,
                [obstacle.r],
                obstacle.direction,
                [obstacle.height]
            ]).astype(np.float32)
            for obstacle in self.obstacles
        ]

        # print(f"Moving obstacles: {[obstacle.shape for obstacle in moving_obstacles]}")  # Debugging print

        # Get relative position to the landing platform (using the assigned destination)
        rel_des_position = (uav.destination - uav.pos).astype(np.float32)

        # Combine everything into the final observation
        obs_dict = {
            "self_state": self_state,
            "other_uav_obs": np.array(other_uav_states),
            "obstacles": np.array(obstacles_states),
            "relative_destination": rel_des_position.astype(np.float32),
        }

        # Print the overall observation shapes
        #for key, value in obs_dict.items():
            #print(f"{key} obs shape: {value.shape}")

        return obs_dict

    def _get_reward(self, uav):
        """
        Calculates the reward for the UAV based on its current state and the environment.

        Reward structure:
        +100 for successfully reaching the destination
        -100 for leading to a collision
        -100 for being out of bounds
        -2 for each time step penalty
        +1 for moving toward the destination
        -1 if action exceeds max acceleration
        -1 if velocity exceeds max velocity
        """
        reward = 0.0

        # Check if the UAV has successfully reached the destination
        is_reached, rel_dist, rel_vel = uav.check_dest_reached()

        if is_reached:
            reward += self.reward_params["destination_reached_reward"]
            uav.done = True  # Mark the UAV as done if it reaches its destination
        else:
            # Time step penalty
            reward += self.reward_params["time_step_penalty"]

            # Check if the UAV is moving towards the destination
            destination_dir = uav.destination - uav.pos
            if np.dot(uav.vel, destination_dir) > 0:
                reward += self.reward_params["towards_dest_reward"]  # Moving towards the destination

        # Check for collisions with other UAVs, moving obstacles, and vertiport cylinders
        uav_collision = any([uav.in_collision(other_uav) for other_uav in self.uavs.values() if uav.id != other_uav.id])
        obstacle_collision = any(
            [uav.in_collision(obstacle) for obstacle in self.obstacles if obstacle.type == ObsType.M])

        if uav_collision or obstacle_collision:
            reward += self.reward_params["collision_penalty"]  # Penalty for collisions
            uav.done = True

        # Check if the UAV is out of bounds
        if self.is_out_of_bounds(uav):
            reward += self.reward_params["out_of_bounds_penalty"]  # Penalty for going out of bounds
            # uav.done = True  # End the UAV's run if it is out of bounds

        # Penalty if the action exceeds maximum acceleration
        if np.linalg.norm(uav.acceleration) > self.max_acceleration:
            reward += self.reward_params["exceed_acc_penalty"]

        # Penalty if the velocity exceeds the maximum allowed velocity
        if np.linalg.norm(uav.vel) > self.max_velocity:
            reward += self.reward_params["exceed_vel_penalty"]

        return reward

    def seed(self, seed=None):
        """Random value to seed"""
        random.seed(seed)
        np.random.seed(seed)

        # Properly seed the environment using Gym's seeding utility
        np_random, seed = seeding.np_random(seed)
        self.np_random = np_random  # Store the seeded RandomState if needed
        return [seed]

    def is_out_of_bounds(self, uav):
        """
        Check if the UAV is out of the defined environment bounds.

        :param uav: The UAV to check.
        :return: True if the UAV is out of bounds, False otherwise.
        """
        return not (
                -self.env_max_w <= uav.pos[0] <= self.env_max_w and
                -self.env_max_l <= uav.pos[1] <= self.env_max_l and
                self.env_min_h <= uav.pos[2] <= self.env_max_h
        )

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to its initial state, including UAVs, obstacles, and vertiport components.
        UAVs will be randomly assigned to start from the grid or from above the platform.
        """
        # Reset environment time
        self._time_elapsed = 0.0
        self.uavs = {}

        self._agent_ids = set(range(self.num_uavs))

        # Create a Grid instance
        self.grid_1 = Grid(grid_size=4, spacing=1.0, z_height=self.env_min_h + 0.5)
        self.grid_2 = Grid(grid_size=4, spacing=1.0, z_height=self.env_max_h - 0.5)


        grid_1_points = self.grid_1.grid_points.copy()
        #print(grid_1_points)

        grid_2_points = self.grid_2.grid_points.copy()
        #print(grid_2_points)

        # Shuffle the platform points to randomly assign UAVs
        np.random.shuffle(grid_1_points)
        np.random.shuffle(grid_2_points)

        # Randomly determine how many UAVs will start from the grid
        num_uav_grid_1_start = np.random.randint(0, self.num_uavs + 1)  # Random number of UAVs from grid
        #print(f"num_uav_grid_1_start: {num_uav_grid_1_start}")
        num_uav_grid_2_start = self.num_uavs - num_uav_grid_1_start  # The rest will start from platform
        #print(f"num_uav_grid_2_start: {num_uav_grid_2_start}")
        # Initialize UAVs, ensuring no collisions at start
        self._initialize_uavs(num_uav_grid_1_start, num_uav_grid_2_start, grid_1_points, grid_2_points)

        #print(self.uavs)
        # Calculate initial observations and rewards
        obs = {uav.id: self._get_obs(uav) for uav in self.uavs.values()}
        reward = {uav.id: self._get_reward(uav) for uav in self.uavs.values()}

        # Gather additional info
        info = {uav.id: self._get_info(uav) for uav in self.uavs.values()}
        #print(self.uavs)
        return obs, info
