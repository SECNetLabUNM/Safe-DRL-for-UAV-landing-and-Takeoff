import sys
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from uav_sim.agents.uav import Obstacle, UAV, ObsType, AgentType, Entity, Cylinder, Platform, Vertiport, Grid
from uav_sim.utils.gui import Gui
from qpsolvers import solve_qp
import logging
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UavSim(MultiAgentEnv):

    def __init__(self, env_config={}):
        super().__init__()
        self.dt = env_config.setdefault("dt", 0.1)
        self._seed = env_config.setdefault("seed", None)
        # self.render_mode = env_config.setdefault("render_mode", "human")
        self.num_uavs = env_config.setdefault("num_uavs", 4)
        self.num_obstacles = env_config.setdefault("num_obstacles", 4)
        self.obstacle_radius = env_config.setdefault("obstacle_radius", 0.1)

        self.gamma = env_config.setdefault("gamma", 1)

        self.max_time = env_config.setdefault("max_time", 50.0)
        self.max_num_obstacles = env_config.setdefault("max_num_obstacles", 6)

        if self.max_num_obstacles < self.num_obstacles:
            self.num_obstacles = self.max_num_obstacles
            raise ValueError(
                f"Max number of obstacles {self.max_num_obstacles} is less than number of obstacles {self.num_obstacles}")
        self.obstacle_collision_weight = env_config.setdefault(
            "obstacle_collision_weight", 1
        )
        self.uav_collision_weight = env_config.setdefault("uav_collision_weight", 1)

        self.max_velocity = env_config.setdefault("max_velocity", 0.5)  # Set max velocity
        self.max_acceleration = env_config.setdefault("max_acceleration", 0.5)  # Set max acceleration

        self._use_safe_action = env_config.get("use_safe_action", False)

        self.reward_params = {
            "destination_reached_reward": env_config.get("destination_reached_reward", 100.0),
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

        self.max_rel_dist = np.linalg.norm(
            [2 * self.env_max_w, 2 * self.env_max_h, self.env_max_h]
        )

        self._z_high = env_config.setdefault("z_high", self.env_max_h)
        self._z_high = min(self.env_max_h, self._z_high)
        self._z_low = env_config.setdefault("z_low", 3.0)
        self._z_low = max(0, self._z_low)

        # Initialize cylinder_params here
        self.cylinder_params = env_config.get("cylinder_params", {
            "platform1": {"x": 1, "y": 0, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "platform2": {"x": -1, "y": 0, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "connector1": {"x": 0, "y": 0, "z": 0.3, "direction": np.array([1, 0, 0]), "height": 2, "r": 0.1},
            "platform3": {"x": 0, "y": 1, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "platform4": {"x": 0, "y": -1, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "connector2": {"x": 0, "y": 0, "z": 0.3, "direction": np.array([0, 1, 0]), "height": 2, "r": 0.1},
            "platform5": {"x": 1, "y": 0, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "platform6": {"x": -1, "y": 0, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "connector3": {"x": 0, "y": 0, "z": 1.4, "direction": np.array([1, 0, 0]), "height": 2, "r": 0.1},
            "platform7": {"x": 0, "y": 1, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "platform8": {"x": 0, "y": -1, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2},
            "connector4": {"x": 0, "y": 0, "z": 1.4, "direction": np.array([0, 1, 0]), "height": 2, "r": 0.1},
            "center_connection": {"x": 0, "y": 0, "z": 0.8, "direction": np.array([0, 0, 1]), "height": 1.6, "r": 0.1},
            "center_base": {"x": 0, "y": 0, "z": 0.05, "direction": np.array([0, 0, 1]), "height": 0.1, "r": 0.4},
        })

        # Create a Grid instance
        self.grid = Grid(grid_size=4, spacing=1.0, z_height=self.env_max_h - 0.5)

        # Initialize platform and vertiport using the existing cylinder parameters
        self.vertiport = Vertiport(self.cylinder_params)

        self._env_config = env_config
        self.norm_action_high = np.ones(3)
        self.norm_action_low = np.ones(3) * -1

        self.action_high = np.ones(3) * 1.0
        self.action_low = -self.action_high

        # self.gui = None
        self._time_elapsed = 0.0
        self.seed(self._seed)

        # Initialize obstacles and UAVs
        self.obstacles = self._initialize_obstacles()
        self.uavs = {}

        self.reset()
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
                    low=self.action_low,
                    high=self.action_high,
                    shape=(3,),
                    dtype=np.float32,
                )

    def _initialize_obstacles(self):
        """Initialize obstacles with random positions."""
        obstacles = []
        for i in range(self.num_obstacles):
            obstacle = Obstacle(
                _id=i,
                x=random.uniform(-self.env_max_w, self.env_max_w),
                y=random.uniform(-self.env_max_l, self.env_max_l),
                z=random.uniform(self._z_low, self._z_high - 1),
                r=self.obstacle_radius,
                _type=ObsType.M,
                # _type=ObsType.M if i % 2 == 0 else ObsType.S,
            )
            obstacles.append(obstacle)
        return obstacles

    def _get_observation_space(self):
        entity_state_shape = 11  # pos (3), vel (3), radius (1), direction (3), height (1)

        self_state_shape = entity_state_shape
        num_other_uavs = self.num_uavs - 1
        other_uav_shape = (num_other_uavs, entity_state_shape)
        #print(f"other_uav_shape: {other_uav_shape}")
        obstacle_shape = (self.num_obstacles, entity_state_shape)
        platform_state_shape = 3  # Platform position only
        vertiport_cylinder_shape = (
        len(self.vertiport.platform.platforms) + len(self.vertiport.connector.connectors), entity_state_shape)
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
                    shape=(platform_state_shape,),
                    dtype=np.float32,
                ),
                "vertiport_cylinders": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=vertiport_cylinder_shape,
                    dtype=np.float32,
                ),
            }
        )
        # Print the overall observation shapes
        #for key, value in obs_space.items():
            #print(f"{key} shape: {value.shape}")

        return obs_space

    def get_h(self, uav, entity):
        # For spheres (UAVs and moving obstacles)
        if entity.type == AgentType.U or entity.type == AgentType.O:
            del_p = uav.pos - entity.pos
            del_v = uav.vel - entity.vel

            # Distance minus the sum of the radii of the two spheres
            h = np.linalg.norm(del_p) - (uav.r + entity.r)
            h = np.sqrt(max(0, h))  # Avoid taking the square root of a negative number
            h += (del_p.T @ del_v) / np.linalg.norm(del_p)

            return h

        elif entity.type == AgentType.C:
            # For cylinders (vertiport components)
            p_ik = uav.pos - entity.pos
            cylinder_axis = entity.direction

            # Projection of the UAV's position onto the cylinder's axis
            p_vert_ik = np.dot(p_ik, cylinder_axis) * cylinder_axis
            p_perp_ik = p_ik - p_vert_ik

            # Distance from the side of the cylinder
            dist_to_side = np.linalg.norm(p_perp_ik) - entity.r
            # Distance from the top or bottom of the cylinder
            dist_to_cap = np.abs(np.linalg.norm(p_vert_ik)) - (entity.l / 2)

            # Combine the side and cap distances to calculate h
            h_side = np.sqrt(max(0, dist_to_side))
            h_cap = np.sqrt(max(0, dist_to_cap))

            # Return the minimum distance (side or cap)
            return min(h_side, h_cap)

    def get_b(self, uav, entity):
        del_p = uav.pos - entity.pos  # Position difference
        del_v = uav.vel - entity.vel  # Velocity difference

        h = self.get_h(uav, entity)  # The h value computed from `get_h`

        # Now compute b following the same logic as in the first code block
        b = self.gamma * h ** 3 * np.linalg.norm(del_p)  # Apply cubic function on h and scale with the distance
        b -= ((del_v.T @ del_p) ** 2) / (
                (np.linalg.norm(del_p)) ** 2)  # Subtract the term related to velocities and position
        b += (del_v.T @ del_p) / (np.maximum(np.linalg.norm(del_p) - (uav.r + entity.r), 1e-3))
        # Add the velocity influence term # Avoid division by zero
        b += np.linalg.norm(del_v) ** 2  # Add velocity squared

        return b

    def get_safe_action(self, uav, des_action):
        """
        Use quadratic programming (QP) to adjust the UAV's action to avoid collisions with spheres and cylinders.

        :param uav: The UAV whose action is being adjusted.
        :param des_action: The desired action for the UAV before collision avoidance is applied.
        :return: A safe action that avoids collisions with other UAVs, obstacles, and cylinders.
        """
        G = []
        h = []
        P = np.eye(3)
        u_in = des_action.copy()
        q = -np.dot(P.T, u_in)

        # Other UAVs and moving obstacles (modeled as spheres)
        for other_uav in self.uavs.values():
            if other_uav.id != uav.id:
                G.append(-(uav.pos - other_uav.pos).T)
                b = self.get_b(uav, other_uav)  # Sphere-to-sphere constraint
                h.append(b)

        for obstacle in self.obstacles:
            if obstacle.type == ObsType.M:  # Only check moving obstacles
                G.append(-(uav.pos - obstacle.pos).T)
                b = self.get_b(uav, obstacle)  # Sphere-to-sphere constraint
                h.append(b)

        # Avoid collisions with cylinders (vertiport components)
        for cylinder in self.vertiport.platform.platforms + self.vertiport.connector.connectors:
            p_ik = uav.pos - cylinder.pos  # Vector from UAV center to cylinder center
            cylinder_axis = cylinder.direction  # Cylinder's axis direction

            # Projection of the UAV's position onto the cylinder's axis
            p_vert_ik = np.dot(p_ik, cylinder_axis) * cylinder_axis
            # Vector perpendicular to the cylinder axis
            p_perp_ik = p_ik - p_vert_ik

            # Check the distance to the sides of the cylinder
            if np.linalg.norm(p_perp_ik) > cylinder.r:
                dist_to_side = np.linalg.norm(p_perp_ik) - cylinder.r
                G.append(-p_perp_ik.T)
                b = dist_to_side  # Side collision avoidance
                h.append(b)

            # Check the distance to the top/bottom of the cylinder
            if np.abs(np.linalg.norm(p_vert_ik)) > (cylinder.height / 2):
                dist_to_cap = np.abs(np.linalg.norm(p_vert_ik)) - (cylinder.height / 2)
                G.append(-p_vert_ik.T)
                b = dist_to_cap  # Top/bottom collision avoidance
                h.append(b)

        G = np.array(G)
        h = np.array(h)

        # Solve the QP problem to find the safe action
        if G.size > 0 and h.size > 0:
            try:
                u_out = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )

            except Exception as e:
                logger.warning(f"QP Solver error: {e}. Using desired action.")
                u_out = des_action
        else:
            logger.debug("No constraints for QP solver, using desired action.")
            u_out = des_action  # Consistent return value

        if u_out is None:
            logger.warning("Infeasible solver, using desired action.")
            u_out = des_action  # Always return a valid action

        return u_out

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

            # Use the safe action function to avoid collisions
            if self._use_safe_action:
                action = self.get_safe_action(self.uavs[i], action)
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

        # Get  individual rewards
        reward = {k: v for k, v in reward.items()}

        # Collect additional information
        info = {
            uav.id: self._get_info(uav)
            for uav in self.uavs.values()
            if uav.id in self.alive_agents
        }

        # Determine if all UAVs have finished (landed or marked as done)
        all_done = all([uav.done for uav in self.uavs.values()])

        # Calculate done flags for each UAV
        done = {
            self.uavs[uav_id].id: self.uavs[uav_id].done or all_done
            for uav_id in self.alive_agents
        }

        # Calculate truncated flags for each UAV
        truncated = {
            self.uavs[uav_id].id: self.time_elapsed >= self.max_time
            for uav_id in self.alive_agents
        }

        # Check if the simulation should be terminated or truncated
        done["__all__"] = all(v for v in done.values()) or self.time_elapsed >= self.max_time
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

        # Check for collisions with vertiport components (cylinders)
        vertiport_collision = any(
            [uav.in_collision(cylinder) for cylinder in
             self.vertiport.platform.platforms + self.vertiport.connector.connectors])

        # Check if UAV is out of the environment bounds
        out_of_bounds = self.is_out_of_bounds(uav)

        # Collect the information
        info = {
            "time_step": self.time_elapsed,  # Current time step
            "uav_collision": uav_collision,  # Collision with other UAVs
            "obstacle_collision": obstacle_collision,  # Collision with moving obstacles
            "vertiport_collision": vertiport_collision,  # Collision with vertiport components
            "uav_rel_dist": rel_dist,  # Distance to destination
            "uav_rel_vel": rel_vel,  # Relative velocity to destination
            "uav_reached_dest": 1.0 if is_reached else 0.0,  # 1 if UAV has reached its destination, 0 otherwise
            "uav_out_of_bounds": 1.0 if out_of_bounds else 0.0  # 1 if UAV is out of bounds, 0 otherwise
        }

        return info

    def _get_closest_obstacles(self, uav, num_to_return=7):
        """
        Returns the closest 'num_to_return' entities (including UAVs, moving obstacles, and vertiport cylinders)
        to the given UAV.

        :param uav: The UAV for which we want to find the closest entities.
        :param num_to_return: The maximum number of closest entities to return (default is 7).
        :return: A list of the closest entities (UAVs, obstacles, and vertiport components).
        """
        # Get the states of all other UAVs
        uav_states = [other_uav.state for other_uav in self.uavs.values() if uav.id != other_uav.id]

        # Get the states of all moving obstacles
        obstacle_states = [obstacle.state for obstacle in self.obstacles if obstacle.type == ObsType.M]

        # Get the states of all cylinders (vertiport components)
        vertiport_cylinder_states = [cylinder.state for cylinder in
                                     (self.vertiport.platform.platforms + self.vertiport.connector.connectors)]

        # Combine all entities (UAVs, obstacles, and cylinders)
        all_entities = [other_uav for other_uav in self.uavs.values() if uav.id != other_uav.id] \
                       + [obstacle for obstacle in self.obstacles if obstacle.type == ObsType.M] \
                       + (self.vertiport.platform.platforms + self.vertiport.connector.connectors)

        # Combine all positions and calculate distances from the current UAV
        all_states = uav_states + obstacle_states + vertiport_cylinder_states
        distances = np.linalg.norm(np.array(all_states)[:, :3] - uav.state[:3][None, :], axis=1)

        # Get the indices of the closest entities sorted by distance
        closest_indices = np.argsort(distances)[:num_to_return]

        # Return the closest entities
        closest_entities = [all_entities[idx] for idx in closest_indices]

        return closest_entities

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

        # Verify the shape and type of direction vector
        assert isinstance(uav.direction, np.ndarray), f"UAV {uav.id} direction is not a numpy array!"
        assert uav.direction.shape == (3,), f"UAV {uav.id} direction shape is incorrect: {uav.direction.shape}"

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
                # print(f"Other UAV {other_uav.id} direction before concat: {other_uav.direction}")  # Debugging direction

                # Validate direction vector for other UAVs
                assert isinstance(other_uav.direction,
                                  np.ndarray), f"Other UAV {other_uav.id} direction is not a numpy array!"
                assert other_uav.direction.shape == (
                    3,), f"Other UAV {other_uav.id} direction shape is incorrect: {other_uav.direction.shape}"

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

        moving_obstacles = np.array(obstacles_states)

        # Get relative position to the landing platform (using the assigned destination)
        rel_des_position = uav.pos - uav.destination

        # Get vertiport components (cylinders) states (position, velocity, radius, direction, height)
        vertiport_cylinders = [
            np.concatenate([
                cylinder.pos,
                np.zeros(3),  # Cylinders are static, so velocity is zero
                [cylinder.r],
                cylinder.direction,
                [cylinder.height]
            ]).astype(np.float32)
            for cylinder in (self.vertiport.platform.platforms + self.vertiport.connector.connectors)
        ]

        # Combine everything into the final observation
        obs_dict = {
            "self_state": self_state,
            "other_uav_obs": np.array(other_uav_states),
            "obstacles": np.array(moving_obstacles),
            "relative_destination": rel_des_position.astype(np.float32),
            "vertiport_cylinders": np.array(vertiport_cylinders)
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
        vertiport_collision = any(
            [uav.in_collision(cylinder) for cylinder in
             (self.vertiport.platform.platforms + self.vertiport.connector.connectors)])

        if uav_collision or obstacle_collision or vertiport_collision:
            reward += self.reward_params["collision_penalty"]  # Penalty for collisions
            uav.done = True

        # Check if the UAV is out of bounds
        if self.is_out_of_bounds(uav):
            reward += self.reward_params["out_of_bounds_penalty"]  # Penalty for going out of bounds
            # uav.done = True  # End the UAV's run if it is out of bounds

        # Penalty if the action exceeds maximum acceleration
        if np.any(np.abs(uav.acceleration) > self.max_acceleration):
            reward += self.reward_params["exceed_acc_penalty"]

        # Penalty if the velocity exceeds the maximum allowed velocity
        if np.any(np.abs(uav.vel) > self.max_velocity):
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

    def get_random_pos(
            self,
            low_h=1,
            x_high=None,
            y_high=None,
            z_high=None,
    ):
        if x_high is None:
            x_high = self.env_max_w
        if y_high is None:
            y_high = self.env_max_l
        if z_high is None:
            z_high = self.env_max_h

        x = np.random.uniform(low=-x_high, high=x_high)
        y = np.random.uniform(low=-y_high, high=y_high)
        z = np.random.uniform(low=low_h, high=z_high)
        return np.array([x, y, z])

    def is_in_collision(self, uav):
        # Only check entities near the UAV by narrowing the search using a distance threshold
        potential_collisions = self._get_closest_obstacles(uav, num_to_return=7)  # or k-d trees for optimization
        collision = any([uav.in_collision(entity) for entity in potential_collisions])
        return collision

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
        self.grid = Grid(grid_size=4, spacing=1.0, z_height=self.env_max_h - 0.5)

        # Initialize platform and vertiport using the existing cylinder parameters
        self.vertiport = Vertiport(self.cylinder_params)

        # Collect points above the platforms as start positions or destinations
        platform_points_start = self.vertiport.platform.destination_info.copy()
        grid_points_start = self.grid.grid_points.copy()

        platform_points_end = self.vertiport.platform.destination_info.copy()
        grid_points_end = self.grid.grid_points.copy()
        # print(platform_points)
        # Shuffle the platform points to randomly assign UAVs
        np.random.shuffle(platform_points_start)
        np.random.shuffle(platform_points_end)

        # Randomly determine how many UAVs will start from the grid
        num_uav_grid_start = np.random.randint(0, self.num_uavs + 1)  # Random number of UAVs from grid
        num_uav_platform_start = self.num_uavs - num_uav_grid_start  # The rest will start from platform

        # Initialize UAVs, ensuring no collisions at start
        # Assign UAVs that start on the grid and move to platform points
        for agent_id in range(num_uav_grid_start):
            # Get a random grid point as the start position
            start_position = grid_points_start.pop()
            # Get a platform point as the destination
            destination = platform_points_end.pop()

            # Initialize the UAV with the grid start position and platform as the destination
            uav = self._uav_type(
                _id=agent_id,
                x=start_position[0],
                y=start_position[1],
                z=start_position[2],
                dt=self.dt,
            )

            # Set the UAV's destination after initialization
            uav.set_new_destination(destination=destination)  # Set destination as needed

            # Set initial distance to destination
            uav.last_rel_dist = np.linalg.norm([uav.x, uav.y, uav.z] - np.array(destination))
            self.uavs[agent_id] = uav

        # Assign UAVs that start at the platform and move to grid points
        for agent_id in range(num_uav_grid_start, self.num_uavs):
            # Get a random platform point as the start position
            start_position = platform_points_start.pop()
            # Get a random grid point as the destination
            destination = grid_points_end.pop()
            # Initialize the UAV with platform start position and grid point as the destination
            uav = self._uav_type(
                _id=agent_id,
                x=start_position[0],
                y=start_position[1],
                z=start_position[2],
                dt=self.dt,
            )
            # Set the UAV's destination after initialization
            uav.set_new_destination(destination=destination)  # Set destination as needed

            # Set initial distance to destination
            uav.last_rel_dist = np.linalg.norm([uav.x, uav.y, uav.z] - np.array(destination))
            self.uavs[agent_id] = uav
        #print(f"Number of remaining start point of grid:{len(grid_points_start)}")
        #print(f"Number of remaining end point of grid:{len(grid_points_end)}")
        #print(f"Number of remaining start point of platform:{len(platform_points_start)}")
        #print(f"Number of remaining end point of platform:{len(platform_points_end)}")

        # print(self.uavs)
        # Calculate initial observations and rewards
        obs = {uav.id: self._get_obs(uav) for uav in self.uavs.values()}
        reward = {uav.id: self._get_reward(uav) for uav in self.uavs.values()}

        # Gather additional info
        info = {uav.id: self._get_info(uav) for uav in self.uavs.values()}

        return obs, info
