import pygame
import sys
import random
import math
import datetime
import os
import csv
import argparse

from constants import (WIDTH, HEIGHT, WHITE, CREATURE_ENERGY_DECAY,
                       CREATURE_RADIUS, CREATURE_MAX_ENERGY,
                       NN_HIDDEN_NODES, NN_INPUT_NODES, NN_OUTPUT_NODES,
                       NN_MUTATION_AMOUNT,
                       BURST_NN_THRESHOLD, BURST_ENERGY_COST, BURST_DURATION_FRAMES, BURST_SPEED_MULTIPLIER,
                       FOOD_ENERGY_GAIN, FOOD_RADIUS, FOOD_COLOR, COLOR_MUTATION_AMOUNT, MUTATION_CHANCE)
from nn import tanh  # Assuming nn.py contains the tanh function


# --- The Creature Class ---
class Creature:
    def __init__(self, x=None, y=None, color=None, energy=None,
                 nn_weights_ih=None, nn_biases_h=None, nn_weights_ho=None, nn_biases_o=None):
        self.x = x if x is not None else random.randint(0, WIDTH)
        self.y = y if y is not None else random.randint(0, HEIGHT)
        self.radius = CREATURE_RADIUS
        self.color = color if color is not None else (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.energy = energy if energy is not None else CREATURE_MAX_ENERGY

        self.speed = 1.5 # Base speed
        self.direction = random.uniform(0, 360)

        self.is_dying = False
        self.fade_alpha = 255
        self.food_eaten_individual = 0

        # --- NEW: Burst State ---
        self.is_bursting = False
        self.burst_frames_left = 0
        self.bursts_activated_individual = 0 # Track individual bursts
        self.energy_spent_bursting_individual = 0 # Track individual energy spent on bursts


        # Neural network initialization
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(NN_HIDDEN_NODES)] for _ in range(NN_INPUT_NODES)] if nn_weights_ih is None else nn_weights_ih
        self.biases_h = [random.uniform(-1, 1) for _ in range(NN_HIDDEN_NODES)] if nn_biases_h is None else nn_biases_h
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(NN_OUTPUT_NODES)] for _ in range(NN_HIDDEN_NODES)] if nn_weights_ho is None else nn_weights_ho
        self.biases_o = [random.uniform(-1, 1) for _ in range(NN_OUTPUT_NODES)] if nn_biases_o is None else nn_biases_o


    def get_sensor_data(self, food_items):
        inputs = []
        inputs.append(self.energy / CREATURE_MAX_ENERGY)

        nearest_food_dist = float('inf')
        nearest_food_angle = 0
        if food_items:
            min_dist = float('inf')
            closest_food = None
            for food in food_items:
                dist = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_food = food

            if closest_food:
                nearest_food_dist = min_dist

                food_vector_x = closest_food.x - self.x
                food_vector_y = closest_food.y - self.y

                creature_dir_x = math.cos(math.radians(self.direction))
                creature_dir_y = math.sin(math.radians(self.direction))

                dot_product = creature_dir_x * food_vector_x + creature_dir_y * food_vector_y
                magnitude_creature = math.sqrt(creature_dir_x**2 + creature_dir_y**2)
                magnitude_food = math.sqrt(food_vector_x**2 + food_vector_y**2)

                if magnitude_creature * magnitude_food != 0:
                    raw_angle = math.acos(max(-1, min(1, dot_product / (magnitude_creature * magnitude_food))))
                    cross_product = creature_dir_x * food_vector_y - creature_dir_y * food_vector_x
                    if cross_product < 0:
                        raw_angle *= -1
                    nearest_food_angle = math.degrees(raw_angle)
                else:
                    nearest_food_angle = 0

        MAX_VIEW_DISTANCE = 300
        normalized_dist = 1.0 - (min(nearest_food_dist, MAX_VIEW_DISTANCE) / MAX_VIEW_DISTANCE)
        inputs.append(normalized_dist)

        normalized_angle = nearest_food_angle / 130.0
        inputs.append(normalized_angle)

        return inputs

    def think(self, inputs):
        hidden_inputs = [0.0] * NN_HIDDEN_NODES
        for i in range(NN_INPUT_NODES):
            for j in range(NN_HIDDEN_NODES):
                hidden_inputs[j] += inputs[i] * self.weights_ih[i][j]
        hidden_outputs = [tanh(hidden_inputs[j] + self.biases_h[j]) for j in range(NN_HIDDEN_NODES)]

        output_inputs = [0.0] * NN_OUTPUT_NODES # This is now 2 outputs
        for i in range(NN_HIDDEN_NODES):
            for j in range(NN_OUTPUT_NODES):
                output_inputs[j] += hidden_outputs[i] * self.weights_ho[i][j]
        final_outputs = [tanh(output_inputs[j] + self.biases_o[j]) for j in range(NN_OUTPUT_NODES)] # Apply tanh to both outputs

        steering_force = final_outputs[0]
        burst_decision_raw = final_outputs[1] # This is the raw output for burst

        return steering_force, burst_decision_raw

    def move(self, food_items):
        sensor_inputs = self.get_sensor_data(food_items)
        steering_force, burst_decision_raw = self.think(sensor_inputs)

        # --- Speed Burst Logic ---
        # If not currently bursting and NN output suggests burst and has enough energy
        if not self.is_bursting and burst_decision_raw > BURST_NN_THRESHOLD and self.energy >= BURST_ENERGY_COST:
            self.energy -= BURST_ENERGY_COST
            self.is_bursting = True
            self.burst_frames_left = BURST_DURATION_FRAMES
            self.bursts_activated_individual += 1 # Increment individual burst count
            self.energy_spent_bursting_individual += BURST_ENERGY_COST # Track individual energy spent

        current_speed = self.speed
        if self.is_bursting:
            current_speed = self.speed * BURST_SPEED_MULTIPLIER
            self.burst_frames_left -= 1
            if self.burst_frames_left <= 0:
                self.is_bursting = False

        # Apply current_speed for movement
        self.direction += steering_force * 7 # Max turn angle
        self.direction = self.direction % 360

        direction_vector = pygame.math.Vector2.from_polar((1, self.direction))
        new_x = self.x + current_speed * direction_vector.x
        new_y = self.y + current_speed * direction_vector.y

        # Boundary collision logic
        if new_x - self.radius < 0:
            self.x = self.radius
            self.direction = 180 - self.direction
        elif new_x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.direction = 180 - self.direction
        else:
            self.x = new_x

        if new_y - self.radius < 0:
            self.y = self.radius
            self.direction = 360 - self.direction
        elif new_y + self.radius > HEIGHT:
            self.y = HEIGHT - self.radius
            self.direction = 360 - self.direction
        else:
            self.y = new_y

        # Energy decay
        if not self.is_dying:
            self.energy -= CREATURE_ENERGY_DECAY


    def draw(self, screen):
        current_color = self.color
        if self.is_dying:
            current_color = (self.color[0], self.color[1], self.color[2], int(self.fade_alpha))
            s = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, current_color, (self.radius, self.radius), self.radius)
            screen.blit(s, (int(self.x - self.radius), int(self.y - self.radius)))
        else:
            pygame.draw.circle(screen, current_color, (int(self.x), int(self.y)), self.radius)
            if self.is_bursting: # Draw a small dot or glow to indicate bursting
                pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius + 2, 1) # Yellow outline


        line_length = self.radius * 1.5
        direction_for_line = pygame.math.Vector2.from_polar((1, self.direction))
        end_x = self.x + line_length * direction_for_line.x
        end_y = self.y + line_length * direction_for_line.y
        pygame.draw.line(screen, WHITE, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)


    def eat_food(self, food_items):
        eaten_food = []
        for food in food_items:
            distance = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if distance < (self.radius + food.radius):
                self.energy += FOOD_ENERGY_GAIN
                if self.energy > CREATURE_MAX_ENERGY:
                    self.energy = CREATURE_MAX_ENERGY
                self.food_eaten_individual += 1
                eaten_food.append(food)
        return eaten_food

    def reproduce(self):
        offspring_color = list(self.color)

        for i in range(3):
            if random.random() < MUTATION_CHANCE:
                offspring_color[i] += random.uniform(-COLOR_MUTATION_AMOUNT, COLOR_MUTATION_AMOUNT)
                offspring_color[i] = max(0, min(255, int(offspring_color[i])))

        offspring_weights_ih = [row[:] for row in self.weights_ih]
        offspring_biases_h = self.biases_h[:]
        offspring_weights_ho = [row[:] for row in self.weights_ho]
        offspring_biases_o = self.biases_o[:]

        for i in range(NN_INPUT_NODES):
            for j in range(NN_HIDDEN_NODES):
                if random.random() < MUTATION_CHANCE:
                    offspring_weights_ih[i][j] += random.uniform(-NN_MUTATION_AMOUNT, NN_MUTATION_AMOUNT)
        for i in range(NN_HIDDEN_NODES):
            if random.random() < MUTATION_CHANCE:
                offspring_biases_h[i] += random.uniform(-NN_MUTATION_AMOUNT, NN_MUTATION_AMOUNT)
        for i in range(NN_HIDDEN_NODES):
            for j in range(NN_OUTPUT_NODES): # Loop over NN_OUTPUT_NODES (now 2)
                if random.random() < MUTATION_CHANCE:
                    offspring_weights_ho[i][j] += random.uniform(-NN_MUTATION_AMOUNT, NN_MUTATION_AMOUNT)
        for i in range(NN_OUTPUT_NODES): # Loop over NN_OUTPUT_NODES (now 2)
            if random.random() < MUTATION_CHANCE:
                offspring_biases_o[i] += random.uniform(-NN_MUTATION_AMOUNT, NN_MUTATION_AMOUNT)

        offspring_x = self.x + random.uniform(-self.radius*2, self.radius*2)
        offspring_y = self.y + random.uniform(-self.radius*2, self.radius*2)

        return Creature(x=offspring_x, y=offspring_y,
                        color=tuple(offspring_color),
                        energy=CREATURE_MAX_ENERGY,
                        nn_weights_ih=offspring_weights_ih,
                        nn_biases_h=offspring_biases_h,
                        nn_weights_ho=offspring_weights_ho,
                        nn_biases_o=offspring_biases_o)
        
        
# --- The Food Class (No changes needed here) ---
class Food:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.radius = FOOD_RADIUS
        self.color = FOOD_COLOR

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)