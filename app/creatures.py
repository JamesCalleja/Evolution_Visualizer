"""
This module defines the classes for creatures in the simulation:
Creature, Food, and Obstacle.
"""

import random
import math
import pygame
import constants # Import constants for properties like RADIUS, ENERGY, COLORS, NN_NODES
from nn import tanh # Import tanh activation function from nn module

# Helper function (extracted from main and placed here as it's used by creatures)
def is_position_safe(pos_x, pos_y, check_radius, obstacles_list):
    """
    Checks if a given position is safe for spawning (not too close to obstacles).

    Args:
        pos_x (float): X-coordinate to check.
        pos_y (float): Y-coordinate to check.
        check_radius (float): The radius around the point to check for collision.
        obstacles_list (list): List of Obstacle objects in the world.

    Returns:
        bool: True if the position is safe, False otherwise.
    """
    if not obstacles_list: # If no obstacles, any position is safe
        return True

    for obs in obstacles_list:
        # Find the closest point on the rectangle to the center of the circle
        closest_x = max(obs.x, min(pos_x, obs.x + obs.width))
        closest_y = max(obs.y, min(pos_y, obs.y + obs.height))

        # Calculate the distance between the circle's center and this closest point
        distance_x = pos_x - closest_x
        distance_y = pos_y - closest_y
        distance_squared = (distance_x * distance_x) + (distance_y * distance_y)

        # If the distance squared is less than (check_radius + MIN_SPAWN_DISTANCE_FROM_OBSTACLE) squared,
        # then it's too close.
        # Ensure MIN_SPAWN_DISTANCE_FROM_OBSTACLE is defined in constants.py
        if distance_squared < (check_radius + constants.MIN_SPAWN_DISTANCE_FROM_OBSTACLE)**2:
            return False
    return True


class Creature:
    """Represents a single creature in the simulation."""
    def __init__(self, x=None, y=None, color=None, energy=None,
                 nn_weights_ih=None, nn_biases_h=None,
                 nn_weights_ho=None, nn_biases_o=None,
                 obstacles_ref=None, nn_hidden_nodes=None):
        """
        Initializes a new creature with given or random properties.

        Args:
            x (float, optional): Initial x-coordinate. If None, random.
            y (float, optional): Initial y-coordinate. If None, random.
            color (tuple, optional): RGB color tuple. If None, random.
            energy (float, optional): Initial energy. If None, defaults to CREATURE_MAX_ENERGY.
            nn_weights_ih (list, optional): Weights for input to hidden layer. If None, random.
            nn_biases_h (list, optional): Biases for hidden layer. If None, random.
            nn_weights_ho (list, optional): Weights for hidden to output layer. If None, random.
            nn_biases_o (list, optional): Biases for output layer. If None, random.
            obstacles_ref (list): Reference to the global list of Obstacle objects.
            nn_hidden_nodes (int, optional): Number of nodes in the hidden layer.
                                             Defaults to constants.NN_HIDDEN_NODES.
        """
        self.radius = constants.CREATURE_RADIUS
        self.obstacles_ref = obstacles_ref # Store the reference to obstacles

        # If no position provided, find a safe random position
        if x is None or y is None:
            safe_pos_found = False
            attempts = 0
            while not safe_pos_found and attempts < 1000: # Limit attempts to avoid infinite loop
                temp_x = random.randint(self.radius, constants.WIDTH - self.radius)
                temp_y = random.randint(self.radius, constants.HEIGHT - self.radius)
                if is_position_safe(temp_x, temp_y, self.radius, self.obstacles_ref):
                    self.x = temp_x
                    self.y = temp_y
                    safe_pos_found = True
                attempts += 1
            if not safe_pos_found: # Fallback if no safe position found after many attempts
                self.x = random.randint(self.radius, constants.WIDTH - self.radius)
                self.y = random.randint(self.radius, constants.HEIGHT - self.radius)
        else:
            self.x = x
            self.y = y

        self.color = color if color is not None else (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        self.energy = energy if energy is not None else constants.CREATURE_MAX_ENERGY
        
        # Initial direction and speed
        self.direction = random.uniform(0, 360) # Angle in degrees
        self.speed = constants.CREATURE_BASE_SPEED # Base speed
        self.current_speed = self.speed # Actual speed, adjusted for bursts

        # Speed burst properties
        self.is_bursting = False
        self.burst_frames_left = 0

        # Individual creature stats for fitness calculation and logging
        self.food_eaten_individual = 0
        self.collisions_individual = 0
        self.bursts_activated_individual = 0
        self.burst_energy_spent_individual = 0

        # Dying animation properties
        self.is_dying = False
        self.fade_alpha = 255 # For fading out

        # Neural Network initialization
        # Use NN_HIDDEN_NODES from constants or the passed argument
        self.nn_hidden_nodes = nn_hidden_nodes if nn_hidden_nodes is not None else constants.NN_HIDDEN_NODES

        # Initialize weights and biases. If not provided, create random ones.
        # Ensure NN_INPUT_NODES and NN_OUTPUT_NODES are defined in constants.py
        self.weights_ih = nn_weights_ih if nn_weights_ih is not None else \
                          [[random.uniform(-1, 1) for _ in range(self.nn_hidden_nodes)]
                           for _ in range(constants.NN_INPUT_NODES)]
        self.biases_h = nn_biases_h if nn_biases_h is not None else \
                        [random.uniform(-1, 1) for _ in range(self.nn_hidden_nodes)]
        self.weights_ho = nn_weights_ho if nn_weights_ho is not None else \
                          [[random.uniform(-1, 1) for _ in range(constants.NN_OUTPUT_NODES)]
                           for _ in range(self.nn_hidden_nodes)]
        self.biases_o = nn_biases_o if nn_biases_o is not None else \
                       [random.uniform(-1, 1) for _ in range(constants.NN_OUTPUT_NODES)]


    def get_sensor_data(self, food_items, all_creatures, obstacles):
        """
        Gathers sensor data for the creature's neural network inputs.
        Now includes nearest food, nearest creature (neighbor), and nearest obstacle data.

        Args:
            food_items (list): A list of Food objects currently in the simulation.
            all_creatures (list): A list of all Creature objects in the simulation.
            obstacles (list): A list of Obstacle objects in the simulation.

        Returns:
            list: A list of normalized sensor data (inputs for the NN).
        """
        inputs = []

        # 1. Self Energy (Normalized)
        inputs.append(self.energy / constants.CREATURE_MAX_ENERGY)

        # 2. Nearest Food Data (Normalized Distance and Angle)
        nearest_food_distance = math.hypot(constants.WIDTH, constants.HEIGHT) + 1 # Initialize with a large value
        nearest_food_angle = 0.0

        if food_items:
            nearest_food = None
            for food in food_items:
                dist = math.hypot(self.x - food.x, self.y - food.y)
                if dist < nearest_food_distance:
                    nearest_food_distance = dist
                    nearest_food = food

            if nearest_food:
                max_possible_distance = math.hypot(constants.WIDTH, constants.HEIGHT)
                normalized_food_distance = nearest_food_distance / max_possible_distance
                inputs.append(normalized_food_distance)

                food_vector_x = nearest_food.x - self.x
                food_vector_y = nearest_food.y - self.y

                creature_heading_x = math.cos(math.radians(self.direction)) # Use self.direction
                creature_heading_y = math.sin(math.radians(self.direction)) # Use self.direction

                dot_product = food_vector_x * creature_heading_x + food_vector_y * creature_heading_y
                magnitude_food = math.hypot(food_vector_x, food_vector_y)
                magnitude_creature_heading = math.hypot(creature_heading_x, creature_heading_y)

                if magnitude_food > 0 and magnitude_creature_heading > 0:
                    clamped_dot_prod_norm = max(-1.0, min(1.0, dot_product / (magnitude_food * magnitude_creature_heading)))
                    angle_rad = math.acos(clamped_dot_prod_norm)

                    cross_product = creature_heading_x * food_vector_y - creature_heading_y * food_vector_x
                    if cross_product < 0: # Food is to the right
                        angle_rad = -angle_rad
                    
                    normalized_food_angle = angle_rad / math.pi # Normalize to -1 to 1 range (for -pi to pi)
                    inputs.append(normalized_food_angle)
                else:
                    inputs.extend([0.0]) # No food or no clear direction, set angle to neutral
            else:
                inputs.extend([0.0, 0.0]) # No food found, default to 0 for distance and angle
        else:
            inputs.extend([0.0, 0.0]) # No food items, default to 0 for distance and angle


        # 3. Nearest Neighbor Data (Normalized Distance, Speed, and Angle)
        nearest_neighbor_distance = math.hypot(constants.WIDTH, constants.HEIGHT) + 1 # Initialize with a large value
        nearest_neighbor = None
        
        for other_creature in all_creatures:
            if other_creature != self: # Don't compare creature to itself
                dist = math.hypot(self.x - other_creature.x, self.y - other_creature.y)
                if dist < nearest_neighbor_distance:
                    nearest_neighbor_distance = dist
                    nearest_neighbor = other_creature
        
        if nearest_neighbor:
            # Normalize distance to neighbor
            max_possible_distance = math.hypot(constants.WIDTH, constants.HEIGHT)
            normalized_neighbor_distance = nearest_neighbor_distance / max_possible_distance
            inputs.append(normalized_neighbor_distance)

            # Normalize neighbor's speed (e.g., relative to max possible speed)
            MAX_POSSIBLE_CREATURE_SPEED = constants.CREATURE_BASE_SPEED * constants.BURST_SPEED_MULTIPLIER
            normalized_neighbor_speed = nearest_neighbor.current_speed / MAX_POSSIBLE_CREATURE_SPEED
            inputs.append(normalized_neighbor_speed)

            # Calculate angle to nearest neighbor relative to creature's heading
            neighbor_vector_x = nearest_neighbor.x - self.x
            neighbor_vector_y = nearest_neighbor.y - self.y

            creature_heading_x = math.cos(math.radians(self.direction))
            creature_heading_y = math.sin(math.radians(self.direction))

            dot_product = neighbor_vector_x * creature_heading_x + neighbor_vector_y * creature_heading_y
            magnitude_neighbor = math.hypot(neighbor_vector_x, neighbor_vector_y)
            magnitude_creature_heading = math.hypot(creature_heading_x, creature_heading_y)

            if magnitude_neighbor > 0 and magnitude_creature_heading > 0:
                clamped_dot_prod_norm = max(-1.0, min(1.0, dot_product / (magnitude_neighbor * magnitude_creature_heading)))
                angle_rad = math.acos(clamped_dot_prod_norm)

                cross_product = creature_heading_x * neighbor_vector_y - creature_heading_y * neighbor_vector_x
                if cross_product < 0: # Neighbor is to the right
                    angle_rad = -angle_rad
                
                normalized_neighbor_angle = angle_rad / math.pi
                inputs.append(normalized_neighbor_angle)
            else:
                inputs.extend([0.0, 0.0]) # No neighbor or no clear direction, set angle to neutral
        else:
            inputs.extend([0.0, 0.0, 0.0]) # No neighbor found, default to 0 for distance, speed, and angle

        # 4. Nearest Obstacle Data (Normalized Distance and Angle)
        nearest_obstacle_distance = math.hypot(constants.WIDTH, constants.HEIGHT) + 1
        nearest_obstacle_angle = 0.0

        if obstacles:
            nearest_obstacle = None
            for obs in obstacles:
                # Calculate distance to closest point on obstacle rectangle
                closest_x = max(obs.x, min(self.x, obs.x + obs.width))
                closest_y = max(obs.y, min(self.y, obs.y + obs.height))
                dist = math.hypot(self.x - closest_x, self.y - closest_y) - self.radius # Distance from creature edge

                if dist < nearest_obstacle_distance:
                    nearest_obstacle_distance = dist
                    nearest_obstacle = obs
            
            if nearest_obstacle:
                max_possible_distance = math.hypot(constants.WIDTH, constants.HEIGHT)
                normalized_obstacle_distance = nearest_obstacle_distance / max_possible_distance
                inputs.append(normalized_obstacle_distance)

                # Vector from creature to obstacle's center (approximation)
                obstacle_center_x = nearest_obstacle.x + nearest_obstacle.width / 2
                obstacle_center_y = nearest_obstacle.y + nearest_obstacle.height / 2
                obstacle_vector_x = obstacle_center_x - self.x
                obstacle_vector_y = obstacle_center_y - self.y

                creature_heading_x = math.cos(math.radians(self.direction))
                creature_heading_y = math.sin(math.radians(self.direction))

                dot_product = obstacle_vector_x * creature_heading_x + obstacle_vector_y * creature_heading_y
                magnitude_obstacle = math.hypot(obstacle_vector_x, obstacle_vector_y)
                magnitude_creature_heading = math.hypot(creature_heading_x, creature_heading_y)

                if magnitude_obstacle > 0 and magnitude_creature_heading > 0:
                    clamped_dot_prod_norm = max(-1.0, min(1.0, dot_product / (magnitude_obstacle * magnitude_creature_heading)))
                    angle_rad = math.acos(clamped_dot_prod_norm)

                    cross_product = creature_heading_x * obstacle_vector_y - creature_heading_y * obstacle_vector_x
                    if cross_product < 0: # Obstacle is to the right
                        angle_rad = -angle_rad
                    
                    normalized_obstacle_angle = angle_rad / math.pi
                    inputs.append(normalized_obstacle_angle)
                else:
                    inputs.extend([0.0]) # No obstacle or no clear direction
            else:
                inputs.extend([0.0, 0.0]) # No obstacle found
        else:
            inputs.extend([0.0, 0.0]) # No obstacles in simulation

        return inputs

    def think(self, inputs):
        """
        Processes sensor inputs through the neural network to determine
        steering force and burst decision.

        Args:
            inputs (list): Normalized sensor inputs.

        Returns:
            tuple: (steering_force, burst_decision_raw)
        """
        # Hidden layer calculation
        hidden_inputs = [0.0] * self.nn_hidden_nodes
        for i in range(constants.NN_INPUT_NODES):
            for j in range(self.nn_hidden_nodes):
                # Ensure indices are within bounds for weights_ih
                if i < len(self.weights_ih) and j < len(self.weights_ih[i]):
                    hidden_inputs[j] += inputs[i] * self.weights_ih[i][j]
        hidden_outputs = [tanh(hidden_inputs[j] + self.biases_h[j])
                          for j in range(self.nn_hidden_nodes)]

        # Output layer calculation
        output_inputs = [0.0] * constants.NN_OUTPUT_NODES
        for i in range(self.nn_hidden_nodes):
            for j in range(constants.NN_OUTPUT_NODES):
                # Ensure indices are within bounds for weights_ho
                if i < len(self.weights_ho) and j < len(self.weights_ho[i]):
                    output_inputs[j] += hidden_outputs[i] * self.weights_ho[i][j]
        final_outputs = [
            tanh(output_inputs[j] + self.biases_o[j])
            for j in range(constants.NN_OUTPUT_NODES)
        ]

        steering_force = final_outputs[0]
        burst_decision_raw = final_outputs[1]

        return steering_force, burst_decision_raw

    def move(self, food_items, obstacles, all_creatures): # Added obstacles and all_creatures
        """
        Updates the creature's position based on NN output, handles
        speed bursts, and checks for collisions with boundaries and obstacles.

        Args:
            food_items (list): List of Food objects.
            obstacles (list): List of Obstacle objects.
            all_creatures (list): List of all Creature objects (for neighbor sensing).
        """
        sensor_inputs = self.get_sensor_data(food_items, all_creatures, obstacles) # Pass all arguments
        steering_force, burst_decision_raw = self.think(sensor_inputs)

        # --- Speed Burst Logic ---
        # If not currently bursting and NN output suggests burst and has enough energy
        if (not self.is_bursting and burst_decision_raw > constants.BURST_NN_THRESHOLD and
                self.energy >= constants.BURST_ENERGY_COST):
            self.energy -= constants.BURST_ENERGY_COST
            self.is_bursting = True
            self.burst_frames_left = constants.BURST_DURATION_FRAMES
            self.bursts_activated_individual += 1
            self.burst_energy_spent_individual += constants.BURST_ENERGY_COST

        self.current_speed = self.speed # Start with base speed
        if self.is_bursting:
            self.current_speed = self.speed * constants.BURST_SPEED_MULTIPLIER
            self.burst_frames_left -= 1
            if self.burst_frames_left <= 0:
                self.is_bursting = False

        # Apply current_speed for movement
        self.direction += steering_force * constants.MAX_TURN_ANGLE # Max turn angle
        self.direction %= 360

        direction_vector = pygame.math.Vector2.from_polar((1, self.direction))
        proposed_x = self.x + self.current_speed * direction_vector.x
        proposed_y = self.y + self.current_speed * direction_vector.y

        # --- Obstacle Collision Logic ---
        collided = False
        for obs in self.obstacles_ref: # Use the stored reference to obstacles
            if self.check_collision_with_obstacle(proposed_x, proposed_y, obs):
                collided = True
                self.energy -= constants.OBSTACLE_PENALTY # Apply penalty
                self.collisions_individual += 1
                self.energy = max(0, self.energy)

                # Bounce off the obstacle by reversing direction
                self.direction = (self.direction + 180) % 360
                break # Only penalize/stop for the first obstacle hit

        if not collided:
            self.x = proposed_x
            self.y = proposed_y
        # If collided, creature stays at its current position or gets nudged back by bounce

        # Boundary collision logic (still applies after obstacle check)
        if self.x - self.radius < 0:
            self.x = self.radius
            self.direction = 180 - self.direction
        elif self.x + self.radius > constants.WIDTH:
            self.x = constants.WIDTH - self.radius
            self.direction = 180 - self.direction

        if self.y - self.radius < 0:
            self.y = self.radius
            self.direction = 360 - self.direction
        elif self.y + self.radius > constants.HEIGHT:
            self.y = constants.HEIGHT - self.radius
            self.direction = 360 - self.direction

        # Energy decay
        if not self.is_dying: # Only decay energy if not in dying state
            self.energy -= constants.CREATURE_ENERGY_DECAY
        
        # Set dying state if energy drops to 0
        if self.energy <= 0 and not self.is_dying:
            self.is_dying = True
            self.fade_alpha = 255 # Start fading from opaque


    def check_collision_with_obstacle(self, pos_x, pos_y, obstacle):
        """
        Checks for circular creature colliding with rectangular obstacle.

        Args:
            pos_x (float): Proposed x-coordinate of the creature.
            pos_y (float): Proposed y-coordinate of the creature.
            obstacle (Obstacle): The obstacle to check collision against.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        # Find the closest point on the rectangle to the center of the circle
        closest_x = max(obstacle.x, min(pos_x, obstacle.x + obstacle.width))
        closest_y = max(obstacle.y, min(pos_y, obstacle.y + obstacle.height))

        # Calculate the distance between the circle's center and this closest point
        distance_x = pos_x - closest_x
        distance_y = pos_y - closest_y
        distance_squared = (distance_x * distance_x) + (distance_y * distance_y)

        return distance_squared < (self.radius * self.radius)


    def draw(self, screen_surface):
        """
        Draws the creature on the screen.

        Args:
            screen_surface (pygame.Surface): The surface to draw on.
        """
        # Calculate color based on energy: interpolate between red (low) and original color (high)
        energy_ratio = self.energy / constants.CREATURE_MAX_ENERGY
        # Ensure energy_ratio is between 0 and 1
        energy_ratio = max(0, min(1, energy_ratio))

        original_r, original_g, original_b = self.color
        current_r = int(original_r * energy_ratio + (255 * (1 - energy_ratio))) # Redder when low energy
        current_g = int(original_g * energy_ratio)
        current_b = int(original_b * energy_ratio)

        display_color = (current_r, current_g, current_b)

        if self.is_dying:
            # Fade out effect
            self.fade_alpha -= 5 # Decrease alpha
            if self.fade_alpha < 0:
                self.fade_alpha = 0 # Ensure it doesn't go below 0
            
            s = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            # Create a color with alpha
            fading_color = (display_color[0], display_color[1], display_color[2], int(self.fade_alpha))
            pygame.draw.circle(s, fading_color, (self.radius, self.radius), self.radius)
            screen_surface.blit(s, (int(self.x - self.radius), int(self.y - self.radius)))
        else:
            pygame.draw.circle(screen_surface, display_color,
                               (int(self.x), int(self.y)), self.radius)
            if self.is_bursting: # Draw a small dot or glow to indicate bursting
                pygame.draw.circle(screen_surface, (255, 255, 0),
                                   (int(self.x), int(self.y)),
                                   self.radius + 2, 1) # Yellow outline


        line_length = self.radius * 1.5
        direction_for_line = pygame.math.Vector2.from_polar((1, self.direction))
        end_x = self.x + line_length * direction_for_line.x
        end_y = self.y + line_length * direction_for_line.y
        pygame.draw.line(screen_surface, constants.WHITE, (int(self.x), int(self.y)),
                         (int(end_x), int(end_y)), 1)

    def update(self, food_items, obstacles, all_creatures): # Added update method
        """
        Updates the creature's state including movement, energy decay,
        and checks for death.
        """
        self.move(food_items, obstacles, all_creatures) # Pass all necessary arguments
        self.eat_food(food_items) # This method will handle food consumption

        # If creature is dying, continue fading out
        if self.is_dying:
            self.fade_alpha -= 5 # Rate of fading
            if self.fade_alpha <= 0:
                self.energy = 0 # Ensure energy is truly zero when fully faded
                # The creature will be removed from the main list in main.py once energy is 0

    def eat_food(self, food_items_list):
        """
        Checks for collision with food items and consumes them,
        increasing creature energy.

        Args:
            food_items_list (list): List of Food objects in the world.

        Returns:
            list: A list of Food objects that were eaten.
        """
        eaten_food = []
        for food in food_items_list:
            distance = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if distance < (self.radius + food.radius):
                self.energy += constants.FOOD_ENERGY_GAIN
                self.energy = min(self.energy, constants.CREATURE_MAX_ENERGY)
                self.food_eaten_individual += 1
                eaten_food.append(food)
        return eaten_food

    def calculate_fitness(self):
        """
        Calculates the creature's fitness based on food eaten,
        collisions, and bursts activated.
        """
        fitness = self.food_eaten_individual * constants.FOOD_ENERGY_GAIN
        fitness -= self.collisions_individual * constants.COLLISION_PENALTY_BREEDING
        fitness += self.bursts_activated_individual * constants.BURST_FITNESS_BONUS
        return max(0, fitness) # Fitness cannot be negative

    def reproduce(self, parent2, mutation_chance, nn_mutation_amount, color_mutation_amount, nn_hidden_nodes, obstacles_ref):
        """
        Creates a new offspring creature based on the current creature's
        genetic material (NN weights/biases and color), with mutations.
        This is a simple asexual reproduction with mutation. For sexual,
        you'd combine genes from two parents.

        Args:
            parent2 (Creature): The second parent for genetic mixing (though currently asexual).
            mutation_chance (float): Chance for a gene to mutate (0-1).
            nn_mutation_amount (float): Amount NN weights/biases can change during mutation.
            color_mutation_amount (int): Amount creature color can change during mutation.
            nn_hidden_nodes (int): Number of hidden nodes for the offspring's NN.
            obstacles_ref (list): Reference to the global list of Obstacle objects.

        Returns:
            Creature: A new Creature object representing the offspring.
        """
        # For simplicity, let's just use self (parent1) for genetic material for now
        # You can implement proper crossover with parent2 later.
        
        offspring_color = list(self.color)

        for i in range(3):
            if random.random() < mutation_chance:
                offspring_color[i] += random.uniform(-color_mutation_amount, color_mutation_amount)
                offspring_color[i] = max(0, min(255, int(offspring_color[i])))

        # Initialize offspring NN weights/biases to potentially new sizes based on constants
        # This part ensures that if NN_HIDDEN_NODES changes, the matrices adapt.
        offspring_weights_ih = [[0.0 for _ in range(nn_hidden_nodes)]
                                for _ in range(constants.NN_INPUT_NODES)]
        offspring_biases_h = [0.0 for _ in range(nn_hidden_nodes)]
        offspring_weights_ho = [[0.0 for _ in range(constants.NN_OUTPUT_NODES)]
                                for _ in range(nn_hidden_nodes)]
        offspring_biases_o = [0.0 for _ in range(constants.NN_OUTPUT_NODES)]

        # Copy and mutate existing weights, adapting to new NN_HIDDEN_NODES if sizes differ
        for i in range(constants.NN_INPUT_NODES):
            for j in range(nn_hidden_nodes):
                if i < len(self.weights_ih) and j < len(self.weights_ih[i]):
                    offspring_weights_ih[i][j] = self.weights_ih[i][j]
                else:
                    offspring_weights_ih[i][j] = random.uniform(-1, 1) # New connection

                if random.random() < mutation_chance:
                    offspring_weights_ih[i][j] += random.uniform(-nn_mutation_amount, nn_mutation_amount)
        
        for i in range(nn_hidden_nodes):
            if i < len(self.biases_h):
                offspring_biases_h[i] = self.biases_h[i]
            else:
                offspring_biases_h[i] = random.uniform(-1, 1) # New bias
            
            if random.random() < mutation_chance:
                offspring_biases_h[i] += random.uniform(-nn_mutation_amount, nn_mutation_amount)

        for i in range(nn_hidden_nodes):
            for j in range(constants.NN_OUTPUT_NODES):
                if i < len(self.weights_ho) and j < len(self.weights_ho[i]):
                    offspring_weights_ho[i][j] = self.weights_ho[i][j]
                else:
                    offspring_weights_ho[i][j] = random.uniform(-1, 1) # New connection

                if random.random() < mutation_chance:
                    offspring_weights_ho[i][j] += random.uniform(-nn_mutation_amount, nn_mutation_amount)
        
        for i in range(constants.NN_OUTPUT_NODES):
            if i < len(self.biases_o):
                offspring_biases_o[i] = self.biases_o[i]
            else:
                offspring_biases_o[i] = random.uniform(-1, 1) # New bias

            if random.random() < mutation_chance:
                offspring_biases_o[i] += random.uniform(-nn_mutation_amount, nn_mutation_amount)

        # Spawn offspring near parent (or at random safe spot)
        offspring_x = self.x + random.uniform(-self.radius*2, self.radius*2)
        offspring_y = self.y + random.uniform(-self.radius*2, self.radius*2)

        # Ensure offspring spawns in a safe spot
        attempts = 0
        while not is_position_safe(offspring_x, offspring_y, self.radius, obstacles_ref) and attempts < 100:
            offspring_x = self.x + random.uniform(-self.radius * 5, self.radius * 5)
            offspring_y = self.y + random.uniform(-self.radius * 5, self.radius * 5)
            attempts += 1
        
        # If after attempts, still not safe, just place randomly
        if not is_position_safe(offspring_x, offspring_y, self.radius, obstacles_ref):
            offspring_x = random.randint(self.radius, constants.WIDTH - self.radius)
            offspring_y = random.randint(self.radius, constants.HEIGHT - self.radius)


        return Creature(x=offspring_x, y=offspring_y,
                        color=tuple(offspring_color),
                        energy=constants.CREATURE_MAX_ENERGY, # Start with full energy
                        nn_weights_ih=offspring_weights_ih,
                        nn_biases_h=offspring_biases_h,
                        nn_weights_ho=offspring_weights_ho,
                        nn_biases_o=offspring_biases_o,
                        obstacles_ref=obstacles_ref,
                        nn_hidden_nodes=nn_hidden_nodes) # Pass nn_hidden_nodes


# --- The Food Class ---
class Food:
    """Represents a single food item in the simulation."""
    def __init__(self, obstacles_ref=None):
        """
        Initializes a new food item at a random position.
        Args:
            obstacles_ref (list): Reference to the global list of Obstacle objects.
        """
        self.radius = constants.FOOD_RADIUS
        self.obstacles_ref = obstacles_ref # Store reference to obstacles

        # Find a safe spawn position for the food
        safe_pos_found = False
        attempts = 0
        while not safe_pos_found and attempts < 1000: # Limit attempts
            temp_x = random.randint(self.radius, constants.WIDTH - self.radius)
            temp_y = random.randint(self.radius, constants.HEIGHT - self.radius)
            if is_position_safe(temp_x, temp_y, self.radius, self.obstacles_ref):
                self.x = temp_x
                self.y = temp_y
                safe_pos_found = True
            attempts += 1
        if not safe_pos_found: # If no safe position found, place anywhere
            self.x = random.randint(self.radius, constants.WIDTH - self.radius)
            self.y = random.randint(self.radius, constants.HEIGHT - self.radius)
            
        self.color = constants.FOOD_COLOR

    def draw(self, screen_surface):
        """
        Draws the food item on the screen.

        Args:
            screen_surface (pygame.Surface): The surface to draw on.
        """
        pygame.draw.circle(screen_surface, self.color,
                           (int(self.x), int(self.y)), self.radius)

    def is_eaten_by(self, creature):
        """
        Checks if this food item is eaten by a creature.
        """
        distance = math.hypot(self.x - creature.x, self.y - creature.y)
        return distance < (self.radius + creature.radius)


# --- Obstacle Class ---
class Obstacle:
    """Represents a rectangular obstacle in the simulation."""
    def __init__(self, x, y, width, height, color=constants.OBSTACLE_COLOR):
        """
        Initializes a new obstacle.

        Args:
            x (int): X-coordinate of the top-left corner.
            y (int): Y-coordinate of the top-left corner.
            width (int): Width of the obstacle.
            height (int): Height of the obstacle.
            color (tuple, optional): RGB color tuple. Defaults to OBSTACLE_COLOR.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.rect = pygame.Rect(x, y, width, height)


    def draw(self, screen_surface):
        """
        Draws the obstacle on the screen.

        Args:
            screen_surface (pygame.Surface): The surface to draw on.
        """
        pygame.draw.rect(screen_surface, self.color, self.rect)

