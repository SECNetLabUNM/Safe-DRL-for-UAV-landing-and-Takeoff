import numpy as np
import unittest
from uav_sim.agents.uav import Entity, AgentType  # Adjust the import according to your setup

class TestEntityDistance(unittest.TestCase):
    def setUp(self):
        # Create test entities with different types and positions
        self.entity1 = Entity(_id=1, x=0, y=0, z=0, direction=np.array([0, 0, 1]), height=2, r=1, _type=AgentType.C)
        self.entity2 = Entity(_id=2, x=3, y=0, z=0, direction=np.array([0, 0, 1]), height=2, r=1, _type=AgentType.C)
        self.entity3 = Entity(_id=3, x=0, y=0, z=5, direction=np.array([0, 0, 1]), height=2, r=1, _type=AgentType.C)
        self.entity4 = Entity(_id=4, x=1, y=0, z=0, direction=np.array([0, 0, 1]), height=2, r=1.5, _type=AgentType.C)
        self.entity5 = Entity(_id=5, x=3, y=4, z=5, direction=np.array([0, 0, 0]), height=0, r=2, _type=AgentType.O)

    def test_distance_to_entity(self):
        # Test distance between two non-overlapping cylinders along the x-axis
        distance = self.entity1.rel_distance(self.entity2)
        self.assertAlmostEqual(distance, 3.0)  # Expected center-to-center distance is 3

        # Test distance between two cylinders where one is directly above the other
        distance = self.entity1.rel_distance(self.entity3)
        self.assertAlmostEqual(distance, 5.0)  # Expected center-to-center distance is 5

    def test_in_collision(self):
        # Test for overlap in collision detection
        self.assertTrue(self.entity1.in_collision(self.entity4))  # Expected to return True as they overlap

        # Test for non-overlap in collision detection
        self.assertFalse(self.entity1.in_collision(self.entity2))  # Expected to return False as they don't overlap

    def test_sphere_collision(self):
        # Test for collision between a cylinder and a sphere
        self.assertFalse(self.entity1.in_collision(self.entity5))  # Expected to return False (no collision)

# Run the tests
if __name__ == '__main__':
    unittest.main()
