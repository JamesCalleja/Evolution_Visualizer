import pygame
import sys
import random
import math
import datetime
import os
import csv
import argparse # Import argparse

# --- 1. Constants and Configuration (Now with defaults that can be overridden) ---
# Define default values for your constants. These will be used if no command-line arguments are provided.
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1000, 700
DEFAULT_FPS = 60

DEFAULT_WHITE = (255, 255, 255)
DEFAULT_BLACK = (0, 0, 0)
DEFAULT_BACKGROUND_COLOR = (20, 20, 30)
DEFAULT_FOOD_COLOR = (100, 255, 100)
DEFAULT_FOOD_RADIUS = 3

DEFAULT_CREATURE_RADIUS = 5
DEFAULT_INITIAL_CREATURE_COUNT = 50
DEFAULT_CREATURE_ENERGY_DECAY = 0.05
DEFAULT_FOOD_ENERGY_GAIN = 40
DEFAULT_MAX_FOOD_COUNT = 500

DEFAULT_MUTATION_CHANCE = 0.02
DEFAULT_NN_MUTATION_AMOUNT = 0.1
DEFAULT_COLOR_MUTATION_AMOUNT = 30

# Neural Network Architecture Constants (these are usually fixed per model)
NN_INPUT_NODES = 3
NN_HIDDEN_NODES = 4
NN_OUTPUT_NODES = 1

# Generation Control Defaults
DEFAULT_GENERATION_LENGTH_FRAMES = 5000
DEFAULT_SELECTION_PERCENTAGE = 0.3
DEFAULT_CREATURE_MAX_ENERGY = 100

# Generation End Metric Defaults
DEFAULT_FOOD_LIMIT_PER_GENERATION = 500

# Logging Configuration Defaults
DEFAULT_LOG_DIRECTORY = "simulation_logs"
DEFAULT_LOG_ENABLED = True
DEFAULT_LIVE_LOG_FILE_NAME = "evolution_live_log.csv"

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evolution Simulator (Pygame).")

parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help=f"Window width (default: {DEFAULT_WIDTH})")
parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help=f"Window height (default: {DEFAULT_HEIGHT})")
parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help=f"Frames per second (default: {DEFAULT_FPS})")

parser.add_argument('--initial_creature_count', type=int, default=DEFAULT_INITIAL_CREATURE_COUNT,
                    help=f"Initial number of creatures (default: {DEFAULT_INITIAL_CREATURE_COUNT})")
parser.add_argument('--creature_energy_decay', type=float, default=DEFAULT_CREATURE_ENERGY_DECAY,
                    help=f"Energy decay per frame (default: {DEFAULT_CREATURE_ENERGY_DECAY})")
parser.add_argument('--food_energy_gain', type=float, default=DEFAULT_FOOD_ENERGY_GAIN,
                    help=f"Energy gained from food (default: {DEFAULT_FOOD_ENERGY_GAIN})")
parser.add_argument('--max_food_count', type=int, default=DEFAULT_MAX_FOOD_COUNT,
                    help=f"Maximum food items in world (default: {DEFAULT_MAX_FOOD_COUNT})")

parser.add_argument('--mutation_chance', type=float, default=DEFAULT_MUTATION_CHANCE,
                    help=f"Chance for a gene to mutate (0-1) (default: {DEFAULT_MUTATION_CHANCE})")
parser.add_argument('--nn_mutation_amount', type=float, default=DEFAULT_NN_MUTATION_AMOUNT,
                    help=f"Amount NN weights/biases can change during mutation (default: {DEFAULT_NN_MUTATION_AMOUNT})")
parser.add_argument('--color_mutation_amount', type=int, default=DEFAULT_COLOR_MUTATION_AMOUNT,
                    help=f"Amount creature color can change during mutation (default: {DEFAULT_COLOR_MUTATION_AMOUNT})")

parser.add_argument('--generation_length_frames', type=int, default=DEFAULT_GENERATION_LENGTH_FRAMES,
                    help=f"Frames per generation (default: {DEFAULT_GENERATION_LENGTH_FRAMES})")
parser.add_argument('--selection_percentage', type=float, default=DEFAULT_SELECTION_PERCENTAGE,
                    help=f"Top percentage of creatures to breed (0-1) (default: {DEFAULT_SELECTION_PERCENTAGE})")
parser.add_argument('--creature_max_energy', type=float, default=DEFAULT_CREATURE_MAX_ENERGY,
                    help=f"Maximum energy for a creature (default: {DEFAULT_CREATURE_MAX_ENERGY})")

parser.add_argument('--food_limit_per_generation', type=int, default=DEFAULT_FOOD_LIMIT_PER_GENERATION,
                    help=f"Food items consumed to end generation early (default: {DEFAULT_FOOD_LIMIT_PER_GENERATION})")

parser.add_argument('--log_enabled', type=int, default=int(DEFAULT_LOG_ENABLED),
                    help=f"Enable logging (0=False, 1=True) (default: {int(DEFAULT_LOG_ENABLED)})")

args = parser.parse_args()

# --- Assign values from parsed arguments to constants ---
WIDTH, HEIGHT = args.width, args.height
FPS = args.fps

WHITE = DEFAULT_WHITE
BLACK = DEFAULT_BLACK
BACKGROUND_COLOR = DEFAULT_BACKGROUND_COLOR
FOOD_COLOR = DEFAULT_FOOD_COLOR
FOOD_RADIUS = DEFAULT_FOOD_RADIUS

CREATURE_RADIUS = DEFAULT_CREATURE_RADIUS
INITIAL_CREATURE_COUNT = args.initial_creature_count # This now takes the parsed value
CREATURE_ENERGY_DECAY = args.creature_energy_decay
FOOD_ENERGY_GAIN = args.food_energy_gain
MAX_FOOD_COUNT = args.max_food_count

MUTATION_CHANCE = args.mutation_chance
NN_MUTATION_AMOUNT = args.nn_mutation_amount
COLOR_MUTATION_AMOUNT = args.color_mutation_amount

GENERATION_LENGTH_FRAMES = args.generation_length_frames
SELECTION_PERCENTAGE = args.selection_percentage
CREATURE_MAX_ENERGY = args.creature_max_energy

FOOD_LIMIT_PER_GENERATION = args.food_limit_per_generation

LOG_DIRECTORY = DEFAULT_LOG_DIRECTORY
LOG_ENABLED = bool(args.log_enabled) # Convert 0/1 back to boolean
LIVE_LOG_FILE_NAME = DEFAULT_LIVE_LOG_FILE_NAME
current_log_filepath = os.path.join(LOG_DIRECTORY, LIVE_LOG_FILE_NAME)


# --- Helper function for Neural Network (Activation Functions) ---
def tanh(x):
    return math.tanh(x)

def sigmoid(x):
    return 1 / (1 + math.exp(-1 * x))

def relu(x):
    return max(0, x)

# --- The Creature Class (No changes needed here) ---
class Creature:
    def __init__(self, x=None, y=None, color=None, energy=None, nn_weights_ih=None, nn_biases_h=None, nn_weights_ho=None, nn_biases_o=None):
        self.x = x if x is not None else random.randint(0, WIDTH)
        self.y = y if y is not None else random.randint(0, HEIGHT)
        self.radius = CREATURE_RADIUS
        self.color = color if color is not None else (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.energy = energy if energy is not None else CREATURE_MAX_ENERGY

        self.speed = 1.5
        self.direction = random.uniform(0, 360)

        self.is_dying = False
        self.fade_alpha = 255
        self.food_eaten_individual = 0

        self.weights_ih = [[random.uniform(-1, 1) for _ in range(NN_HIDDEN_NODES)] for _ in range(NN_INPUT_NODES)] if nn_weights_ih is None else nn_weights_ih
        self.biases_h = [random.uniform(-1, 1) for _ in range(NN_HIDDEN_NODES)] if nn_biases_h is None else nn_biases_h
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(NN_OUTPUT_NODES)] for _ in range(NN_HIDDEN_NODES)] if nn_weights_ho is None else nn_weights_ho
        self.biases_o = [random.uniform(-1, 1) for _ in range(NN_HIDDEN_NODES)] if nn_biases_o is None else nn_biases_o # FIX: Should be NN_OUTPUT_NODES

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

        output_inputs = [0.0] * NN_OUTPUT_NODES
        for i in range(NN_HIDDEN_NODES):
            for j in range(NN_OUTPUT_NODES):
                output_inputs[j] += hidden_outputs[i] * self.weights_ho[i][j]
        final_outputs = [tanh(output_inputs[j] + self.biases_o[j]) for j in range(NN_OUTPUT_NODES)]

        return final_outputs[0]

    def move(self, food_items):
        inputs = self.get_sensor_data(food_items)
        steering_force = self.think(inputs)

        MAX_TURN_ANGLE = 7
        self.direction += steering_force * MAX_TURN_ANGLE
        self.direction = self.direction % 360

        direction_vector = pygame.math.Vector2.from_polar((1, self.direction))
        new_x = self.x + self.speed * direction_vector.x
        new_y = self.y + self.speed * direction_vector.y

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
            for j in range(NN_OUTPUT_NODES):
                if random.random() < MUTATION_CHANCE:
                    offspring_weights_ho[i][j] += random.uniform(-NN_MUTATION_AMOUNT, NN_MUTATION_AMOUNT)
        for i in range(NN_OUTPUT_NODES):
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
                        nn_biases_o=offspring_biases_o) # FIX: Should be NN_OUTPUT_NODES


# --- The Food Class (No changes needed here) ---
class Food:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.radius = FOOD_RADIUS
        self.color = FOOD_COLOR

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


# --- 2. Initialize Pygame ---
pygame.init()

# The screen size will now be determined by the arguments passed or defaults
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SRCALPHA)
pygame.display.set_caption("Evolution Simulator Visualizer (NN)")
font = pygame.font.Font(None, 24)
large_font = pygame.font.Font(None, 48)
clock = pygame.time.Clock()

# --- 3. Game State ---
creatures = []
food_items = []
food_eaten_count = 0
food_eaten_this_generation = 0
current_generation = 0
frame_count_this_generation = 0

def create_initial_population(count):
    new_population = []
    for _ in range(count):
        new_population.append(Creature())
    return new_population

# This line MUST come AFTER INITIAL_CREATURE_COUNT is defined (which it now is, from args)
creatures = create_initial_population(INITIAL_CREATURE_COUNT)

# --- Logging Setup ---
log_file = None
log_writer = None
if LOG_ENABLED:
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    log_file = open(current_log_filepath, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "Generation",
        "Food_Eaten_Gen",
        "Top_Creature_Food_Eaten",
        "Num_Survivors",
        "Population_Size_Start_Gen",
        "Avg_Survivor_Energy",
        "Frames_This_Gen"
    ])
    print(f"Logging to: {current_log_filepath}")
else:
    print("Logging is disabled.")


# --- 4. Main Game Loop ---
running = True
simulation_active = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False # This will break the loop and proceed to pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r: # 'R' key now restarts the entire simulation
                print("--- RESTARTING FULL SIMULATION ---")
                if log_file:
                    log_file.close()

                # Reset all game state variables
                creatures = create_initial_population(INITIAL_CREATURE_COUNT)
                food_items = []
                food_eaten_count = 0
                food_eaten_this_generation = 0
                current_generation = 0
                frame_count_this_generation = 0
                simulation_active = True

                # Re-initialize logging for the new run (overwriting the file)
                if LOG_ENABLED:
                    log_file = open(current_log_filepath, 'w', newline='') # Overwrite
                    log_writer = csv.writer(log_file)
                    log_writer.writerow([
                        "Generation",
                        "Food_Eaten_Gen",
                        "Top_Creature_Food_Eaten",
                        "Num_Survivors",
                        "Population_Size_Start_Gen",
                        "Avg_Survivor_Energy",
                        "Frames_This_Gen"
                    ])
                    print(f"Logging to: {current_log_filepath}")

    # --- Update Game State (Evolution Logic) ---
    if simulation_active:
        frame_count_this_generation += 1

        # --- Generational Cycle Logic ---
        if frame_count_this_generation >= GENERATION_LENGTH_FRAMES or food_eaten_this_generation >= FOOD_LIMIT_PER_GENERATION:
            # --- Log data before advancing generation ---
            if LOG_ENABLED and log_writer:
                living_creatures_for_log = [c for c in creatures if not c.is_dying]
                living_creatures_for_log.sort(key=lambda c: c.food_eaten_individual, reverse=True)
                
                top_creature_food = living_creatures_for_log[0].food_eaten_individual if living_creatures_for_log else 0
                
                num_survivors_for_log = max(1, int(len(living_creatures_for_log) * SELECTION_PERCENTAGE))
                survivors_for_log_list = living_creatures_for_log[:num_survivors_for_log]

                avg_survivor_energy = sum(c.energy for c in survivors_for_log_list) / len(survivors_for_log_list) if survivors_for_log_list else 0

                log_writer.writerow([
                    current_generation,
                    food_eaten_this_generation,
                    top_creature_food,
                    len(survivors_for_log_list),
                    len(creatures),
                    f"{avg_survivor_energy:.2f}",
                    frame_count_this_generation
                ])
                log_file.flush()

            current_generation += 1
            frame_count_this_generation = 0

            # 1. Selection
            living_creatures_for_selection = [c for c in creatures if not c.is_dying]
            living_creatures_for_selection.sort(key=lambda c: c.food_eaten_individual, reverse=True)

            num_survivors = max(1, int(len(living_creatures_for_selection) * SELECTION_PERCENTAGE))
            survivors = living_creatures_for_selection[:num_survivors]

            new_generation_creatures = []
            if not survivors:
                print(f"Gen {current_generation-1}: No survivors from previous generation. Re-initializing population.")
                new_generation_creatures = create_initial_population(INITIAL_CREATURE_COUNT)
            else:
                # Add survivors directly to the new generation to ensure their genes persist
                # new_generation_creatures.extend(survivors) # Optional: directly carry over survivors
                
                # Fill remaining spots with offspring from survivors
                for _ in range(INITIAL_CREATURE_COUNT): # Ensure fixed population size
                    parent = random.choice(survivors)
                    offspring = parent.reproduce()
                    if offspring:
                        new_generation_creatures.append(offspring)
                        if len(new_generation_creatures) >= INITIAL_CREATURE_COUNT:
                            break

            creatures = []
            creatures.extend(new_generation_creatures)
            food_items = [] # Clear food for a clean start

            food_eaten_this_generation = 0

            print(f"--- Generation {current_generation} started. Population: {len(creatures)}. Top Breeder Food: {survivors[0].food_eaten_individual if survivors else 'N/A'} ---")


        # Spawn new food if below max limit
        if len(food_items) < MAX_FOOD_COUNT:
            if random.random() < 0.1:
                food_items.append(Food())

        # Update creatures and handle interactions
        creatures_to_remove_after_fade = []

        for creature in list(creatures):
            creature.move(food_items)

            eaten = creature.eat_food(food_items)
            for food in eaten:
                if food in food_items:
                    food_items.remove(food)
                    food_eaten_count += 1
                    food_eaten_this_generation += 1

            if creature.energy <= 0 and not creature.is_dying:
                creature.is_dying = True
                creature.speed = 0

            if creature.is_dying:
                creature.fade_alpha -= 5
                if creature.fade_alpha <= 0:
                    creatures_to_remove_after_fade.append(creature)

        for creature in creatures_to_remove_after_fade:
            if creature in creatures:
                creatures.remove(creature)


    # --- Drawing/Rendering ---
    screen.fill(BACKGROUND_COLOR)

    for food in food_items:
        food.draw(screen)

    for creature in creatures:
        creature.draw(screen)

    # --- UI/Info Panel ---
    population_text = font.render(f"Population: {len(creatures)}", True, WHITE)
    screen.blit(population_text, (10, 10))
    food_in_world_text = font.render(f"Food in World: {len(food_items)}", True, WHITE)
    screen.blit(food_in_world_text, (10, 40))
    total_eaten_text = font.render(f"Food Eaten (Total): {food_eaten_count}", True, WHITE)
    screen.blit(total_eaten_text, (10, 70))
    gen_eaten_text = font.render(f"Food Eaten (Gen): {food_eaten_this_generation}/{FOOD_LIMIT_PER_GENERATION}", True, WHITE)
    screen.blit(gen_eaten_text, (10, 100))
    gen_text = font.render(f"Generation: {current_generation}", True, WHITE)
    screen.blit(gen_text, (10, 130))

    # --- Update the Display ---
    pygame.display.flip()

    # --- Control Frame Rate ---
    clock.tick(FPS)

# --- 5. Quit Pygame ---
pygame.quit()
if log_file:
    log_file.close()
sys.exit() # Important for the subprocess to fully exit