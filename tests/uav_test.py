import numpy as np

# Import the necessary modules from your UAV simulation
from uav_sim.agents.uav import Entity, UAV, Obstacle, Grid, Platform, Vertiport, Cylinder


def test_entity_creation_and_state_access():
    print("Running test: test_entity_creation_and_state_access")
    entity = Entity(_id=1, x=2, y=3, z=4, vx=1, vy=2, vz=3)
    print(f"Entity {id}: Position: {entity.pos}, Velocity: {entity.vel}, Direction: {entity.direction}, Height: {entity.height}, Radius: {entity.r}")
    assert np.array_equal(entity.pos, [2, 3, 4]), "Position not initialized correctly"
    assert np.array_equal(entity.vel, [1, 2, 3]), "Velocity not initialized correctly"
    print("------------------------------------------------------------------------------")


def test_collision_detection():
    print("Running test: test_collision_detection")
    uav1 = UAV(_id=1, x=0, y=0, z=0)
    uav2 = UAV(_id=2, x=0.1, y=0.1, z=0.1, r=0.1)
    print(f"Collision test between UAV1 (Position: {uav1.pos}, Radius: {uav1.r}) and UAV2 (Position: {uav2.pos}, Radius: {uav2.r})")
    assert uav1.in_collision(uav2), "Collision detection failed for close UAVs"

    obstacle1 = Obstacle(_id=3, x=5, y=5, z=0, r=0.1)
    print(f"Collision test between UAV1 (Position: {uav1.pos}, Radius: {uav1.r}) and Obstacle1 (Position: {obstacle1.pos}, Radius: {obstacle1.r})")
    assert not uav1.in_collision(obstacle1), "UAV should NOT collide with nearby obstacle"

    obstacle2 = Obstacle(_id=4, x=0.2, y=0.2, z=0.1, r=0.1)
    print(f"Collision test between UAV1 (Position: {uav1.pos}, Radius: {uav1.r}) and Obstacle2 (Position: {obstacle2.pos}, Radius: {obstacle2.r})")
    assert uav1.in_collision(obstacle2), "UAV should collide with nearby obstacle"

    # Test collision between a UAV and a Cylinder
    cylinder = Cylinder(_id=5, x=0, y=0, z=0, direction=np.array([0, 0, 1]), height=1, r=0.2)
    uav3 = UAV(_id=6, x=0.0, y=0.0, z=0, r=0.1)
    uav4 = UAV(_id=7, x=0.0, y=0.0, z=0.5, r=0.1)
    uav5 = UAV(_id=8, x=0.25, y=0.0, z=0, r=0.1)
    print(f"Collision test between UAV3 (Position: {uav3.pos}, Radius: {uav3.r}) and Cylinder (Position: {cylinder.pos}, Height: {cylinder.height}, Radius: {cylinder.r})")
    assert uav3.in_collision(cylinder), "UAV should collide with the cylinder"

    print(f"Collision test between UAV4 (Position: {uav4.pos}, Radius: {uav4.r}) and Cylinder (Position: {cylinder.pos}, Height: {cylinder.height}, Radius: {cylinder.r})")
    assert uav4.in_collision(cylinder), "UAV should collide with the cylinder"

    print(f"Collision test between UAV5 (Position: {uav5.pos}, Radius: {uav5.r}) and Cylinder (Position: {cylinder.pos}, Height: {cylinder.height}, Radius: {cylinder.r})")
    assert uav5.in_collision(cylinder), "UAV should collide with the cylinder"
    print("------------------------------------------------------------------------------")


def test_uav_movement():
    print("Running test: test_uav_movement")
    uav = UAV(_id=1, x=0, y=0, z=0, vx=0, vy=0, vz=0, r=0.1, dt=1)
    initial_pos = np.copy(uav.pos)
    initial_vel = np.copy(uav.vel)
    input_accleration = np.array([3, 3, 3])
    print(f"Initial State -- Position: {uav.pos}, Velocity: {uav.vel}, Direction: {uav.direction}")

    uav.step(input_accleration)
    print(f"After Movement -- Position: {uav.pos}, Velocity: {uav.vel}, Direction: {uav.direction}")

    # Checking if the new position is correct
    expected_position = initial_pos + initial_vel * uav.dt + 0.5 * input_accleration * uav.dt ** 2
    assert np.allclose(uav.pos, expected_position, atol=0.1), "UAV position did not move as expected"
    assert np.allclose(uav.vel, initial_vel + input_accleration, atol=0.1), "UAV velocity did not update as expected"
    print("------------------------------------------------------------------------------")


def test_obstacle_movement():
    print("Running test: test_obstacle_movement")
    obstacle = Obstacle(_id=1, x=1, y=1, z=1, r=0.1, dt=1)
    initial_pos = np.copy(obstacle.pos)
    initial_vel = np.copy(obstacle.vel)
    input_accleration = np.array([3, 3, 3])
    print(f"Initial State -- Position: {obstacle.pos}, Velocity: {obstacle.vel}, Direction: {obstacle.direction}")

    obstacle.step()
    print(f"After Movement -- Position: {obstacle.pos}, Velocity: {obstacle.vel}, Direction: {obstacle.direction}")
    print("------------------------------------------------------------------------------")


def test_uav_collision_with_movement():
    print("Running test: test_uav_collision_with_movement")
    uav = UAV(_id=1, x=0, y=0, z=0, r=0.1, dt=1)
    obstacle = Obstacle(_id=2, x=0.8, y=0, z=0, r=0.1, dt=1)
    print(f"Initial positions -- UAV (Position: {uav.pos}, Radius: {uav.r}), Obstacle (Position: {obstacle.pos}, Radius: {obstacle.r})")
    assert not uav.in_collision(obstacle), "False positive in collision detection"

    uav.step(np.array([1, 0, 0]))
    print(f"After movement -- UAV (Position: {uav.pos}, Radius: {uav.r}), Obstacle (Position: {obstacle.pos}, Radius: {obstacle.r})")
    assert uav.in_collision(obstacle), "UAV should collide with nearby obstacle"
    print("------------------------------------------------------------------------------")


def test_destination_reach():
    print("Running test: test_destination_reach")
    platform = Platform({"platform1": {"x": 0, "y": 0, "z": 0, "direction": np.array([0, 0, 1]), "height": 1, "r": 0.2}})
    uav = UAV(_id=1, x=0, y=0, z=1.6, dt=1, platform=platform)
    uav.set_new_destination(destination_index=0)
    uav.step(np.array([0, 0, -1]))  # Large step to simulate movement
    print(f"UAV position: {uav.pos}, Destination: {uav.destination}, Radius: {uav.r}")
    assert uav.check_dest_reached(), "UAV did not recognize it reached the destination"
    print("------------------------------------------------------------------------------")


def test_vertiport_initialization():
    print("Running test: test_vertiport_initialization")
    vertiport_params = {
        "platform1": {"x": 0, "y": 0, "z": 0, "direction": np.array([0, 0, 1]), "height": 1, "r": 0.5},
        "connector1": {"x": 0, "y": 1, "z": 0, "direction": np.array([0, 0, 1]), "height": 1, "r": 0.1}
    }
    vertiport = Vertiport(vertiport_params)
    print(f"Vertiport has {len(vertiport.platform.platforms)} platforms and {len(vertiport.connector.connectors)} connectors with details:")
    for platform in vertiport.platform.platforms:
        print(f"Platform ID: {platform.id}, Position: {platform.pos}, Height: {platform.height}, Radius: {platform.r}")
    for connector in vertiport.connector.connectors:
        print(f"Connector ID: {connector.id}, Position: {connector.pos}, Height: {connector.height}, Radius: {connector.r}")
    print("------------------------------------------------------------------------------")


def test_grid_functionality():
    print("Running test: test_grid_functionality")
    grid = Grid(grid_size=4, spacing=1.0, z_height=0.0)
    point = grid.get_random_point()
    print(f"Random grid point: {point}")
    assert point is not None, "Failed to get a random point from the grid"
    assert len(grid.grid_points) == 15, "Grid point not properly removed after access"
    print("------------------------------------------------------------------------------")

def run_tests():
    test_entity_creation_and_state_access()
    test_collision_detection()
    test_uav_movement()
    test_obstacle_movement()
    test_uav_collision_with_movement()
    test_destination_reach()
    test_vertiport_initialization()
    test_grid_functionality()
    print("All tests passed successfully!")


run_tests()
