# main.py
import pygame
import csv
import os
import sys
import argparse
import random

# Import components from your new modules
import constants
from nn import tanh
from creatures import Creature, Food, Obstacle, is_position_safe
from utils import initialize_obstacles

# --- Argument Parsing ---
try:
    print("DEBUG: Parsing arguments...")
    parser = argparse.ArgumentParser(description="Evolution Simulator (Pygame).")

    parser.add_argument('--initial_creature_count', type=int, default=constants.INITIAL_CREATURE_COUNT,
                        help='Initial number of creatures in the simulation.')
    parser.add_argument('--creature_energy_decay', type=float, default=constants.CREATURE_ENERGY_DECAY,
                        help='Amount of energy creatures lose per frame.')
    parser.add_argument('--food_energy_gain', type=float, default=constants.FOOD_ENERGY_GAIN,
                        help='Energy gained by a creature when consuming food.')
    parser.add_argument('--max_food_count', type=int, default=constants.MAX_FOOD_COUNT,
                        help='Maximum number of food items in the world at any time.')
    parser.add_argument('--creature_max_energy', type=float, default=constants.CREATURE_MAX_ENERGY,
                        help='Maximum energy a creature can have.')
    parser.add_argument('--mutation_chance', type=float, default=constants.MUTATION_CHANCE,
                        help='Chance for a gene to mutate during reproduction (0-1).')
    parser.add_argument('--nn_mutation_amount', type=float, default=constants.NN_MUTATION_AMOUNT,
                        help='Amount neural network weights/biases can change during mutation.')
    parser.add_argument('--color_mutation_amount', type=int, default=constants.COLOR_MUTATION_AMOUNT,
                        help='Amount creature color can change during mutation.')
    parser.add_argument('--nn_hidden_nodes', type=int, default=constants.NN_HIDDEN_NODES,
                        help='Number of nodes in the neural network\'s hidden layer.')
    parser.add_argument('--selection_percentage', type=float, default=constants.SELECTION_PERCENTAGE,
                        help='Top percentage of creatures to breed (0-1).')
    parser.add_argument('--generation_length_frames', type=int, default=constants.GENERATION_LENGTH_FRAMES,
                        help='Number of frames each generation lasts.')
    parser.add_argument('--food_limit_per_generation', type=int, default=constants.FOOD_LIMIT_PER_GENERATION,
                        help='Food items consumed to end generation early.')
    parser.add_argument('--width', type=int, default=constants.WIDTH,
                        help='Width of the simulation window.')
    parser.add_argument('--height', type=int, default=constants.HEIGHT,
                        help='Height of the simulation window.')
    parser.add_argument('--fps', type=int, default=constants.FPS,
                        help='Frames per second for the simulation display.')
    parser.add_argument('--log_enabled', type=int, default=1,
                        help='Enable (1) or disable (0) logging to CSV.')

    args = parser.parse_args()
    print("DEBUG: Arguments parsed successfully.")

except Exception as e:
    print(f"ERROR: Argument parsing failed: {e}", file=sys.stderr)
    sys.exit(1)

# --- Assign values from parsed arguments to local variables/constants ---
try:
    print("DEBUG: Assigning constants from arguments...")
    WIDTH = args.width
    HEIGHT = args.height
    FPS = args.fps
    INITIAL_CREATURE_COUNT = args.initial_creature_count
    CREATURE_ENERGY_DECAY = args.creature_energy_decay
    FOOD_ENERGY_GAIN = args.food_energy_gain
    MAX_FOOD_COUNT = args.max_food_count
    CREATURE_MAX_ENERGY = args.creature_max_energy
    MUTATION_CHANCE = args.mutation_chance
    NN_MUTATION_AMOUNT = args.nn_mutation_amount
    COLOR_MUTATION_AMOUNT = args.color_mutation_amount
    NN_HIDDEN_NODES = args.nn_hidden_nodes
    SELECTION_PERCENTAGE = args.selection_percentage
    GENERATION_LENGTH_FRAMES = args.generation_length_frames
    FOOD_LIMIT_PER_GENERATION = args.food_limit_per_generation
    LOG_ENABLED = bool(args.log_enabled)
    print("DEBUG: Constants assigned.")
except Exception as e:
    print(f"ERROR: Failed to assign constants: {e}", file=sys.stderr)
    sys.exit(1)


# --- Pygame Initialization ---
try:
    print("DEBUG: Initializing Pygame...")
    pygame.init()
    print("Pygame initialized successfully!") # This print is now inside a try block
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    print("DEBUG: Pygame display and clock set up.")
except Exception as e:
    print(f"ERROR: Pygame initialization failed: {e}", file=sys.stderr)
    pygame.quit()
    sys.exit(1)


# --- Global simulation lists ---
creatures = []
food_items = []
obstacles = []

# --- Simulation state variables ---
current_generation = 0
frames_this_generation = 0
food_eaten_this_generation = 0
running = True

# --- Logging Setup ---
LOG_DIRECTORY = "simulation_logs"
LIVE_LOG_FILE_NAME = "evolution_live_log.csv"
log_filepath = os.path.join(LOG_DIRECTORY, LIVE_LOG_FILE_NAME)
log_file = None
log_writer = None

if LOG_ENABLED:
    try:
        print("DEBUG: Setting up logging...")
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        log_file = open(log_filepath, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "Generation", "Frames_This_Gen", "Food_Eaten_Gen", "Num_Survivors",
            "Population_Size_Start_Gen", "Avg_Survivor_Energy",
            "Top_Creature_Food_Eaten_Individual", "Top_Creature_Energy_Individual",
            "Top_Creature_Collisions_Individual", "Top_Creature_Bursts_Activated_Individual",
            "Total_Bursts_Activated_Gen", "Total_Burst_Energy_Spent_Gen",
            "Total_Collisions_Gen", "Top_Creature_Combined_Fitness"
        ])
        print("DEBUG: Logging setup complete.")
    except Exception as e:
        print(f"ERROR: Setting up logging failed: {e}", file=sys.stderr)
        LOG_ENABLED = False


# --- Initialize Obstacles (using function from utils) ---
try:
    print("DEBUG: Initializing obstacles...")
    obstacles = initialize_obstacles(constants.NUM_OBSTACLES) # Call without argument based on current utils.py
    print(f"DEBUG: {len(obstacles)} obstacles initialized.")
except Exception as e:
    print(f"ERROR: Obstacle initialization failed: {e}", file=sys.stderr)
    pygame.quit()
    sys.exit(1)


# --- Simulation Functions ---
def spawn_creatures(count):
    new_creatures = []
    for _ in range(count):
        new_creatures.append(Creature(obstacles_ref=obstacles))
    return new_creatures

def spawn_food(count):
    new_food = []
    for _ in range(count):
        new_food.append(Food(obstacles_ref=obstacles))
    return new_food

def reproduce_and_evolve():
    global creatures, current_generation, frames_this_generation, food_eaten_this_generation

    for creature in creatures:
        pass

    creatures.sort(key=lambda c: c.calculate_fitness(), reverse=True)

    if LOG_ENABLED and log_writer:
        num_survivors = len(creatures)
        avg_survivor_energy = sum(c.energy for c in creatures) / num_survivors if num_survivors > 0 else 0
        
        top_creature = creatures[0] if creatures else None
        top_food_eaten = top_creature.food_eaten_individual if top_creature else 0
        top_energy = top_creature.energy if top_creature else 0
        top_collisions = top_creature.collisions_individual if top_creature else 0
        top_bursts = top_creature.bursts_activated_individual if top_creature else 0
        top_fitness = top_creature.calculate_fitness() if top_creature else 0

        total_bursts_gen = sum(c.bursts_activated_individual for c in creatures) if creatures else 0
        total_burst_energy_spent_gen = sum(c.burst_energy_spent_individual for c in creatures) if creatures else 0
        total_collisions_gen = sum(c.collisions_individual for c in creatures) if creatures else 0

        log_writer.writerow([
            current_generation, frames_this_generation, food_eaten_this_generation, num_survivors,
            INITIAL_CREATURE_COUNT, avg_survivor_energy,
            top_food_eaten, top_energy, top_collisions, top_bursts,
            total_bursts_gen, total_burst_energy_spent_gen, total_collisions_gen,
            top_fitness
        ])
        log_file.flush()

    selection_size = max(1, int(len(creatures) * SELECTION_PERCENTAGE))
    parents = creatures[:selection_size]

    next_generation_creatures = []
    for _ in range(INITIAL_CREATURE_COUNT):
        if parents:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = parent1.reproduce(parent2, MUTATION_CHANCE, NN_MUTATION_AMOUNT, COLOR_MUTATION_AMOUNT, NN_HIDDEN_NODES, obstacles_ref=obstacles)
            next_generation_creatures.append(child)
        else:
            next_generation_creatures.append(Creature(obstacles_ref=obstacles))


    creatures = next_generation_creatures
    current_generation += 1
    frames_this_generation = 0
    food_eaten_this_generation = 0
    food_items.clear()
    food_items.extend(spawn_food(MAX_FOOD_COUNT))

    print(f"Generation {current_generation} ended. New generation spawned with {len(creatures)} creatures.")
    print(f"Top creature fitness: {top_fitness:.2f} (Food: {top_food_eaten}, Collisions: {top_collisions}, Bursts: {top_bursts})")


# --- Initial Spawns ---
try:
    print("DEBUG: Spawning initial creatures...")
    creatures.extend(spawn_creatures(INITIAL_CREATURE_COUNT))
    print(f"DEBUG: {len(creatures)} initial creatures spawned.")

    print("DEBUG: Spawning initial food items...")
    food_items.extend(spawn_food(MAX_FOOD_COUNT))
    print(f"DEBUG: {len(food_items)} initial food items spawned.")

    print(f"Initial population: {len(creatures)} creatures, {len(food_items)} food items.")
except Exception as e:
    print(f"ERROR: Initial spawning failed: {e}", file=sys.stderr)
    pygame.quit()
    sys.exit(1)


# --- Game Loop ---
try:
    print("DEBUG: Entering main game loop...")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frames_this_generation += 1

        # Update creatures
        # Create a copy of creatures list to iterate over if elements are removed
        creatures_to_update = list(creatures)
        for creature in creatures_to_update:
            if creature.energy <= 0:
                if creature in creatures: # Check if it hasn't been removed by another creature's interaction
                    creatures.remove(creature)
                continue # Skip update for dead creatures

            creature.update(food_items, obstacles, creatures) # Pass all_creatures for neighbor sensing
            # If creature died during update, it will be removed in next iteration or next frame

        # Remove eaten food and spawn new
        food_items_eaten_this_frame = 0
        food_items_remaining = []
        for food in food_items:
            eaten = False
            for creature in creatures: # Iterate over current creatures list
                if food.is_eaten_by(creature):
                    food_eaten_this_generation += 1
                    food_items_eaten_this_frame += 1
                    eaten = True
                    break
            if not eaten:
                food_items_remaining.append(food)
        food_items = food_items_remaining

        if len(food_items) < MAX_FOOD_COUNT:
            num_to_spawn = MAX_FOOD_COUNT - len(food_items)
            food_items.extend(spawn_food(num_to_spawn))


        # Check for end of generation
        if frames_this_generation >= GENERATION_LENGTH_FRAMES or \
           food_eaten_this_generation >= FOOD_LIMIT_PER_GENERATION or \
           len(creatures) == 0:
            print(f"DEBUG: Generation {current_generation} ending. Reason: Frames ({frames_this_generation}), Food Eaten ({food_eaten_this_generation}), Creatures Left ({len(creatures)})")
            reproduce_and_evolve()
            if not creatures:
                print("All creatures died. Simulation ended.")
                running = False
                break

        # --- Drawing ---
        screen.fill(constants.BACKGROUND_COLOR)

        for food in food_items:
            food.draw(screen)

        for creature in creatures:
            creature.draw(screen)

        for obs in obstacles:
            obs.draw(screen)

        # Display info
        text_surface = font.render(f"Generation: {current_generation}", True, constants.WHITE)
        screen.blit(text_surface, (10, 10))
        text_surface = font.render(f"Creatures: {len(creatures)}", True, constants.WHITE)
        screen.blit(text_surface, (10, 30))
        text_surface = font.render(f"Food Eaten (Gen): {food_eaten_this_generation}/{FOOD_LIMIT_PER_GENERATION}", True, constants.WHITE)
        screen.blit(text_surface, (10, 50))
        text_surface = font.render(f"Frames (Gen): {frames_this_generation}/{GENERATION_LENGTH_FRAMES}", True, constants.WHITE)
        screen.blit(text_surface, (10, 70))
        text_surface = font.render(f"FPS: {clock.get_fps():.0f}", True, constants.WHITE)
        screen.blit(text_surface, (WIDTH - 80, 10))

        pygame.display.flip()
        clock.tick(FPS)

except Exception as e:
    print(f"ERROR: An error occurred in the main game loop: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr) # Print full traceback to stderr
    running = False # Ensure loop terminates

finally:
    # --- Cleanup ---
    print("DEBUG: Cleaning up...")
    if log_file:
        log_file.close()
        print("DEBUG: Log file closed.")
    pygame.quit()
    print("Pygame exited.")
    sys.exit(0) # Exit cleanly