"""
This module contains various utility functions for the simulation,
such as initialization helpers.
"""

import random
import constants # Import constants for OBSTACLE_SIZE, NUM_OBSTACLES, WIDTH, HEIGHT
from creatures import Obstacle # Corrected: Import Obstacle class from creatures.py

def initialize_obstacles(num_obstacles): # <-- CHANGE THIS LINE to accept num_obstacles
    """
    Creates a list of randomly placed square obstacles.

    Args:
        num_obstacles (int): The number of obstacles to create.

    Returns:
        list: A list of Obstacle objects.
    """
    obs = []
    for _ in range(num_obstacles): # <-- AND CHANGE THIS LINE to use num_obstacles
        # Ensure obstacles are placed fully within bounds
        x = random.randint(0, constants.WIDTH - constants.OBSTACLE_SIZE)
        y = random.randint(0, constants.HEIGHT - constants.OBSTACLE_SIZE)
        obs.append(Obstacle(x, y, constants.OBSTACLE_SIZE, constants.OBSTACLE_SIZE))
    return obs

# Note: The `is_position_safe` function has been moved into `creatures.py`
# because it is primarily used during the initialization of `Creature` and `Food` objects.
# If it were used more broadly outside of entity creation, it might fit better here.