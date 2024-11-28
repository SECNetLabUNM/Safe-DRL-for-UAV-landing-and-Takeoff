import unittest
import numpy as np
from uav_sim.agents.uav import Entity, Obstacle, UAV, Grid, AgentType, ObsType

class TestEntityClasses(unittest.TestCase):
    def setUp(self):
        # Initialize some default entities to test
        self.entity = Entity(_id=1, x=1.0, y=2.0, z=3.0, vx=0.5, vy=0.5, vz=0.5)
        self.obstacle = Obstacle(_id=2, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, r=0.1, dt=0.1, _type=ObsType.M)
        self.uav = UAV(_id=3, x=2.0, y=2.0, z=2.0, vx=0.0, vy=0.0, vz=0.0, r=0.1, dt=0.1)
        self.grid = Grid(grid_size=4, spacing=1.0, z_height=1.0)

    def test_entity_state(self):
        # Test state initialization and state properties
        print("______________________________________________________________")
        print("Testing entity state initialization...")
        self.assertTrue(np.allclose(self.entity.state, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(self.entity.pos, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(self.entity.vel, [0.5, 0.5, 0.5]))
        print(f"Entity state: {self.entity.state}, Position: {self.entity.pos}, Velocity: {self.entity.vel}")

    def test_entity_update_state(self):
        # Test the update state function
        print("______________________________________________________________")
        print("Testing entity state update...")
        new_position = np.array([4.0, 5.0, 6.0])
        new_velocity = np.array([0.1, 0.1, 0.1])
        self.entity.update_state(new_position, new_velocity)
        self.assertTrue(np.allclose(self.entity.state, [4.0, 5.0, 6.0, 0.1, 0.1, 0.1]))
        print(f"Updated Entity state: {self.entity.state}")

    def test_entity_collision(self):
        # Test collision detection
        print("______________________________________________________________")
        print("Testing entity collision detection...")
        another_entity = Entity(_id=4, x=1.05, y=2.0, z=3.0, r=0.1, _type=AgentType.U)
        distance = self.entity.rel_distance(another_entity)
        print(f"Entity 1 Position: {self.entity.pos}, Entity 2 Position: {another_entity.pos}")
        print(f"Relative distance between entities: {distance}")
        in_collision = self.entity.in_collision(another_entity)
        print(f"Entities in collision: {in_collision}")
        self.assertTrue(in_collision)

    def test_entity_rel_distance(self):
        # Test relative distance function
        print("______________________________________________________________")
        print("Testing relative distance calculation...")
        another_entity = Entity(_id=5, x=4.0, y=5.0, z=6.0)
        distance = self.entity.rel_distance(another_entity)
        self.assertAlmostEqual(distance, np.linalg.norm([3.0, 3.0, 3.0]))
        print(f"Relative distance between entities: {distance}")

    def test_obstacle_select_random_destination(self):
        # Test if obstacle can select a random destination
        print("______________________________________________________________")
        print("Testing obstacle selecting a random destination...")
        self.obstacle.select_random_destination()
        destination = self.obstacle.destination
        self.assertGreaterEqual(destination[0], -5)
        self.assertLessEqual(destination[0], 5)
        self.assertGreaterEqual(destination[1], -5)
        self.assertLessEqual(destination[1], 5)
        self.assertGreaterEqual(destination[2], 1)
        self.assertLessEqual(destination[2], 4)
        print(f"Obstacle random destination: {destination}")

    def test_obstacle_step(self):
        # Test obstacle movement towards destination
        print("______________________________________________________________")
        print("Testing obstacle movement...")
        self.obstacle.select_random_destination()
        initial_pos = np.copy(self.obstacle.pos)
        self.obstacle.step()
        self.assertFalse(np.array_equal(initial_pos, self.obstacle.pos))
        print(f"Initial position: {initial_pos}, New position: {self.obstacle.pos}")

    def test_uav_step(self):
        # Test UAV step with zero acceleration
        print("______________________________________________________________")
        print("Testing UAV step with zero acceleration...")
        initial_pos = np.copy(self.uav.pos)
        print(f"Initial UAV Position: {initial_pos}, Velocity: {self.uav.vel}")
        self.uav.step(input_acceleration=np.zeros(3))
        print(f"UAV Position after zero acceleration: {self.uav.pos}, Velocity: {self.uav.vel}")
        self.assertTrue(np.allclose(initial_pos, self.uav.pos))

        # Test UAV step with acceleration
        print("Testing UAV step with non-zero acceleration...")
        self.uav.step(input_acceleration=np.array([1.0, 0.0, 0.0]))
        print(f"UAV Position after acceleration: {self.uav.pos}, Velocity: {self.uav.vel}")
        self.assertFalse(np.array_equal(initial_pos, self.uav.pos))

    def test_uav_check_dest_reached(self):
        # Set destination and check if UAV reached
        print("______________________________________________________________")
        print("Testing UAV destination reached...")
        self.uav.set_new_destination(destination=np.array([2.0, 2.0, 2.0]))
        print(f"UAV Current Position: {self.uav.pos}, Destination: {self.uav.destination}")
        reached, _, _ = self.uav.check_dest_reached()
        print(f"Destination reached: {reached}")
        self.assertTrue(reached)

    def test_grid_creation(self):
        # Test grid creation with correct number of points
        print("______________________________________________________________")
        print("Testing grid creation...")
        self.assertEqual(len(self.grid.grid_points), 16)
        expected_point = [-(self.grid.grid_size - 1) / 2 * self.grid.spacing, -(self.grid.grid_size - 1) / 2 * self.grid.spacing, self.grid.z_height]
        self.assertTrue(any(np.allclose(point, expected_point) for point in self.grid.grid_points))
        print(f"Grid points created: {self.grid.grid_points}")

    def test_grid_get_random_point(self):
        # Test if grid can return a random point and remove it from grid_points
        print("______________________________________________________________")
        print("Testing grid random point selection...")
        initial_length = len(self.grid.grid_points)
        point = self.grid.get_random_point()
        self.assertEqual(len(self.grid.grid_points), initial_length - 1)
        self.assertNotIn(point, self.grid.grid_points)
        print(f"Random point selected: {point}, Remaining grid points: {len(self.grid.grid_points)}")

    def test_grid_reset(self):
        # Test grid reset to ensure all points are recreated
        print("______________________________________________________________")
        print("Testing grid reset...")
        self.grid.get_random_point()
        self.grid.reset()
        self.assertEqual(len(self.grid.grid_points), 16)
        print(f"Grid points after reset: {self.grid.grid_points}")

if __name__ == '__main__':
    unittest.main()
