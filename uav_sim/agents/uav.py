import numpy as np
from enum import IntEnum


class AgentType(IntEnum):
    U = 0  # uav
    O = 1  # obstacle


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving


class Entity:
    def __init__(self, _id, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, r=0.1, _type=AgentType.O,
                 direction=np.array([0.0, 0.0, 0.0]), height=0.0, collision_delta=0.1):
        self.id = _id
        self.type = _type
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.r = r
        self.direction = direction  # 3-bit vector representing direction
        self.height = height  # height, for sphere it's 0
        self.collision_delta = collision_delta  # Safe distance buffer for collision
        # x, y, z, x_dot, y_dot, z_dot
        self._state = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz,], dtype=np.float32)

    @property
    def state(self):
        return self._state

    @property
    def pos(self):
        return self._state[0:3]

    @property
    def vel(self):
        return self._state[3:6]

    def update_state(self, position, velocity):
        """Updates position and velocity in state."""
        self._state[:3] = position
        self._state[3:6] = velocity

    def in_collision(self, other_entity):
        """
        Check if this UAV is in collision with another entity (UAV, mobile NCE, or static NCE).

        :param other_entity: The other entity (UAV, mobile NCE, or static NCE)
        :return: True if in collision, False otherwise
        """
        if other_entity.type == AgentType.U:
            # Collision between two UAVs (both modeled as spheres)
            dist = np.linalg.norm(self.pos - other_entity.pos)
            return dist <= (self.r + other_entity.r + self.collision_delta)

        elif other_entity.type == AgentType.O:
            # Collision between UAV and mobile NCE (also modeled as a sphere)
            dist = np.linalg.norm(self.pos - other_entity.pos)
            return dist <= (self.r + other_entity.r + self.collision_delta)

        return False

    def rel_distance(self, entity=None):
        if entity is None:
            return np.linalg.norm(self.pos)

        return np.linalg.norm(self.pos - entity.pos)

    def rel_vel(self, entity=None):
        if entity is None:
            return np.linalg.norm(self.vel)

        return np.linalg.norm(self.vel - entity.vel)

    def get_closest_entities(self, entity_list, num_to_return=None):
        num_entities = len(entity_list)

        # return empty list if list is empty
        if num_entities == 0:
            return []

        if num_to_return is None:
            num_to_return = num_entities
        else:
            num_to_return = max(0, min(num_entities, num_to_return))

        entity_states = np.array([entity.state for entity in entity_list])

        dist = np.linalg.norm(entity_states[:, :3] - self._state[:3][None, :], axis=1)
        argsort = np.argsort(dist)[:num_to_return]
        closest_entities = [entity_list[idx] for idx in argsort]

        return closest_entities


class Obstacle(Entity):
    def __init__(self, _id, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, r=0.1, dt=0.1, _type=ObsType.M, max_velocity=0.25, collision_delta=0.1):
        super().__init__(_id, x, y, z, vx, vy, vz, r, _type=AgentType.O, direction=np.array([0.0, 0.0, 0.0]), height=0.0, collision_delta=collision_delta)
        self.dt = dt
        self.destination = np.array([x, y, z])
        self.max_velocity = max_velocity  # Set max velocity as a scalar

    def select_random_destination(self):
        """Randomly select a new destination, ensuring it's sufficiently far from the current position."""
        while True:
            new_destination = np.array([
                np.random.uniform(-5, 5),  # x range
                np.random.uniform(-5, 5),  # y range
                np.random.uniform(1, 4)  # z range
            ])
            if np.linalg.norm(new_destination - self.pos) > 1.0:
                self.destination = new_destination
                break

    def step(self):
        # Small epsilon value to avoid division by zero
        epsilon = 1e-5

        if self.type == ObsType.M:
            # Calculate the distance to the destination
            direction = self.destination - self.pos
            #print(f"Position: {self.pos}, Velocity: {self.vel}")
            distance = np.linalg.norm(direction)
            #print(f"distance:{distance}")
            # If reached destination, select a new random destination
            if distance <= 0.1:
                self.select_random_destination()
                direction = self.destination - self.pos
                distance = np.linalg.norm(direction)
            #print(f"direction:{direction}")
            #print(f"distance:{distance}")
            # Normalize direction and move towards the destination
            # Adding epsilon to avoid division by zero
            velocity = self.max_velocity * (direction / (distance + epsilon))
            #print(f"velocity:{velocity}")
            # Update the state of the obstacle

            new_position = self.pos + velocity * self.dt
            self.update_state(new_position, velocity)
            #print(f"State -- Position: {self.pos}, Velocity: {self.vel}")
        else:
            # Fixed obstacles do nothing
            pass


class UAV(Entity):
    def __init__(self, _id, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, r=0.1, dt=0.1, collision_delta=0.1):
        super().__init__(_id, x, y, z, vx, vy, vz, r, _type=AgentType.U, direction=np.array([0.0, 0.0, 0.0]), height=0.0, collision_delta=collision_delta)
        self.dt = dt  # Time step
        self.acceleration = np.zeros(3)  # Acceleration along x, y, z
        self.destination = None  # UAV's current target destination
        self.done = False  # Initialize the done attribute
        # Set the initial destination if not provided
        if self.destination is None:
            self.destination = np.array([0.0, 0.0, 0.0])

    def step(self, input_acceleration=np.zeros(3)):
        """Updates the UAV's position and velocity using the provided acceleration input."""
        # Clip the input acceleration to the maximum allowed acceleration
        acceleration = input_acceleration

        # Update velocity using V' = V + a * t
        new_velocity = self.vel + acceleration * self.dt

        # Update position using P' = P + (V + V') * t / 2
        new_position = self._state[:3] + (self.vel + new_velocity) * self.dt / 2

        # Update state
        self.update_state(new_position, new_velocity)

    def check_dest_reached(self):
        """Check if the UAV has reached its destination."""
        if self.destination is None:
            return False

        # Compute the distance to the current destination
        rel_dist = np.linalg.norm(self.pos - self.destination)

        #print(f"Reltive distance: {rel_dist}")
        # Consider the destination reached if the UAV is within a small threshold distance (e.g., 0.1 units)
        epsilon = 1e-6
        return rel_dist <= (0.1 + epsilon), rel_dist, self.vel

    def set_acceleration(self, ax, ay, az):
        """Set the UAV's acceleration along x, y, z axes."""
        self.acceleration = np.array([ax, ay, az])

    def set_new_destination(self, destination=None, destination_index=0):
        """Sets a new destination, either provided or from the platform."""
        if destination is not None:
            self.destination = destination
        else:
            self.destination = None

class Grid:
    def __init__(self, grid_size=4, spacing=1.0, z_height=0.0):
        """
        Initializes a grid in the x-y plane with the specified number of points and spacing.
        :param grid_size: The size of the grid (e.g., 4x4)
        :param spacing: The spacing between grid points
        :param z_height: The z-coordinate for the grid (height)
        """
        self.grid_size = grid_size
        self.spacing = spacing
        self.z_height = z_height
        self.grid_points = self.create_grid()

    def create_grid(self):
        """Create grid points in the x-y plane centered at (0, 0)."""
        grid = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = (i - (self.grid_size - 1) / 2) * self.spacing
                y = (j - (self.grid_size - 1) / 2) * self.spacing
                grid.append([x, y, self.z_height])
        return grid

    def get_random_point(self):
        """Randomly return a point from the grid if available."""
        if self.grid_points:
            return self.grid_points.pop(np.random.randint(0, len(self.grid_points)))
        else:
            raise ValueError("No spots left in the grid.")

    def reset(self):
        """Reset the grid points and return the new grid."""
        self.grid_points = self.create_grid()
        return self.grid_points