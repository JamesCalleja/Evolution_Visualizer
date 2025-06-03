import pygame
import csv
import os
import sys
import argparse
import random

# Import components from your new modules
import constants as const
from nn import tanh
from creatures import Creature, Food 

# from creatures import Creature, Food, Obstacle, is_position_safe
# from utils import initialize_obstacles

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evolution Simulator (Pygame).")

parser.add_argument('--width', type=int, default=const.DEFAULT_WIDTH, help=f"Window width (default: {const.DEFAULT_WIDTH})")
parser.add_argument('--height', type=int, default=const.DEFAULT_HEIGHT, help=f"Window height (default: {const.DEFAULT_HEIGHT})")
parser.add_argument('--fps', type=int, default=const.DEFAULT_FPS, help=f"Frames per second (default: {const.DEFAULT_FPS})")

parser.add_argument('--initial_creature_count', type=int, default=const.DEFAULT_INITIAL_CREATURE_COUNT,
                    help=f"Initial number of creatures (default: {const.DEFAULT_INITIAL_CREATURE_COUNT})")
parser.add_argument('--creature_energy_decay', type=float, default=const.DEFAULT_CREATURE_ENERGY_DECAY,
                    help=f"Energy decay per frame (default: {const.DEFAULT_CREATURE_ENERGY_DECAY})")
parser.add_argument('--food_energy_gain', type=float, default=const.DEFAULT_FOOD_ENERGY_GAIN,
                    help=f"Energy gained from food (default: {const.DEFAULT_FOOD_ENERGY_GAIN})")
parser.add_argument('--max_food_count', type=int, default=const.DEFAULT_MAX_FOOD_COUNT,
                    help=f"Maximum food items in world (default: {const.DEFAULT_MAX_FOOD_COUNT})")

parser.add_argument('--mutation_chance', type=float, default=const.DEFAULT_MUTATION_CHANCE,
                    help=f"Chance for a gene to mutate (0-1) (default: {const.DEFAULT_MUTATION_CHANCE})")
parser.add_argument('--nn_mutation_amount', type=float, default=const.DEFAULT_NN_MUTATION_AMOUNT,
                    help=f"Amount NN weights/biases can change during mutation (default: {const.DEFAULT_NN_MUTATION_AMOUNT})")
parser.add_argument('--color_mutation_amount', type=int, default=const.DEFAULT_COLOR_MUTATION_AMOUNT,
                    help=f"Amount creature color can change during mutation (default: {const.DEFAULT_COLOR_MUTATION_AMOUNT})")

parser.add_argument('--generation_length_frames', type=int, default=const.DEFAULT_GENERATION_LENGTH_FRAMES,
                    help=f"Frames per generation (default: {const.DEFAULT_GENERATION_LENGTH_FRAMES})")
parser.add_argument('--selection_percentage', type=float, default=const.DEFAULT_SELECTION_PERCENTAGE,
                    help=f"Top percentage of creatures to breed (0-1) (default: {const.DEFAULT_SELECTION_PERCENTAGE})")
parser.add_argument('--creature_max_energy', type=float, default=const.DEFAULT_CREATURE_MAX_ENERGY,
                    help=f"Maximum energy for a creature (default: {const.DEFAULT_CREATURE_MAX_ENERGY})")

parser.add_argument('--food_limit_per_generation', type=int, default=const.DEFAULT_FOOD_LIMIT_PER_GENERATION,
                    help=f"Food items consumed to end generation early (default: {const.DEFAULT_FOOD_LIMIT_PER_GENERATION})")

parser.add_argument('--log_enabled', type=int, default=int(const.DEFAULT_LOG_ENABLED),
                    help=f"Enable logging (0=False, 1=True) (default: {int(const.DEFAULT_LOG_ENABLED)})")

args = parser.parse_args()


pygame.init()

screen = pygame.display.set_mode((const.WIDTH, const.HEIGHT), pygame.SRCALPHA)
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

# --- NEW: Global counters for burst data per generation ---
total_bursts_activated_gen = 0
total_burst_energy_spent_gen = 0


def create_initial_population(count):
    new_population = []
    for _ in range(count):
        new_population.append(Creature())
    return new_population

creatures = create_initial_population(const.INITIAL_CREATURE_COUNT)

# --- Logging Setup ---
log_file = None
log_writer = None
if const.LOG_ENABLED:
    if not os.path.exists(const.LOG_DIRECTORY):
        os.makedirs(const.LOG_DIRECTORY)
    log_file = open(const.log_filepath, 'w', newline='')
    log_writer = csv.writer(log_file)
    # --- UPDATED CSV HEADER ---
    log_writer.writerow([
        "Generation",
        "Food_Eaten_Gen",
        "Top_Creature_Food_Eaten",
        "Num_Survivors",
        "Population_Size_Start_Gen",
        "Avg_Survivor_Energy",
        "Total_Bursts_Activated_Gen", # NEW
        "Total_Burst_Energy_Spent_Gen", # NEW
        "Frames_This_Gen"
    ])
    print(f"Logging to: {const.log_filepath}")
else:
    print("Logging is disabled.")


# --- 4. Main Game Loop ---
running = True
simulation_active = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.K_r: # 'R' key now restarts the entire simulation
                print("--- RESTARTING FULL SIMULATION ---")
                if log_file:
                    log_file.close()

                # Reset all game state variables
                creatures = create_initial_population(const.INITIAL_CREATURE_COUNT)
                food_items = []
                food_eaten_count = 0
                food_eaten_this_generation = 0
                current_generation = 0
                frame_count_this_generation = 0
                simulation_active = True

                # --- NEW: Reset burst counters on restart ---
                total_bursts_activated_gen = 0
                total_burst_energy_spent_gen = 0

                # Re-initialize logging for the new run (overwriting the file)
                if const.LOG_ENABLED:
                    log_file = open(const.log_filepath, 'w', newline='') # Overwrite
                    log_writer = csv.writer(log_file)
                    # --- UPDATED CSV HEADER for restart ---
                    log_writer.writerow([
                        "Generation",
                        "Food_Eaten_Gen",
                        "Top_Creature_Food_Eaten",
                        "Num_Survivors",
                        "Population_Size_Start_Gen",
                        "Avg_Survivor_Energy",
                        "Total_Bursts_Activated_Gen", # NEW
                        "Total_Burst_Energy_Spent_Gen", # NEW
                        "Frames_This_Gen"
                    ])
                    print(f"Logging to: {const.log_filepath}")


    # --- Update Game State (Evolution Logic) ---
    if simulation_active:
        frame_count_this_generation += 1

        # --- Generational Cycle Logic ---
        if frame_count_this_generation >= const.GENERATION_LENGTH_FRAMES or food_eaten_this_generation >= const.FOOD_LIMIT_PER_GENERATION:
            # --- Log data before advancing generation ---
            if const.LOG_ENABLED and log_writer:
                living_creatures_for_log = [c for c in creatures if not c.is_dying]
                living_creatures_for_log.sort(key=lambda c: c.food_eaten_individual, reverse=True)

                top_creature_food = living_creatures_for_log[0].food_eaten_individual if living_creatures_for_log else 0

                num_survivors_for_log = max(1, int(len(living_creatures_for_log) * const.SELECTION_PERCENTAGE))
                survivors_for_log_list = living_creatures_for_log[:num_survivors_for_log]

                avg_survivor_energy = sum(c.energy for c in survivors_for_log_list) / len(survivors_for_log_list) if survivors_for_log_list else 0

                log_writer.writerow([
                    current_generation,
                    food_eaten_this_generation,
                    top_creature_food,
                    len(survivors_for_log_list),
                    len(creatures),
                    f"{avg_survivor_energy:.2f}",
                    total_bursts_activated_gen, # NEW
                    total_burst_energy_spent_gen, # NEW
                    frame_count_this_generation
                ])
                log_file.flush()

            current_generation += 1
            frame_count_this_generation = 0

            # 1. Selection
            living_creatures_for_selection = [c for c in creatures if not c.is_dying]
            living_creatures_for_selection.sort(key=lambda c: c.food_eaten_individual, reverse=True)

            num_survivors = max(1, int(len(living_creatures_for_selection) * const.SELECTION_PERCENTAGE))
            survivors = living_creatures_for_selection[:num_survivors]

            new_generation_creatures = []
            if not survivors:
                print(f"Gen {current_generation-1}: No survivors from previous generation. Re-initializing population.")
                new_generation_creatures = create_initial_population(const.INITIAL_CREATURE_COUNT)
            else:
                # Fill remaining spots with offspring from survivors
                for _ in range(const.INITIAL_CREATURE_COUNT): # Ensure fixed population size
                    parent = random.choice(survivors)
                    offspring = parent.reproduce()
                    if offspring:
                        new_generation_creatures.append(offspring)
                        if len(new_generation_creatures) >= const.INITIAL_CREATURE_COUNT:
                            break

            creatures = []
            creatures.extend(new_generation_creatures)
            food_items = [] # Clear food for a clean start

            food_eaten_this_generation = 0
            # --- NEW: Reset burst counters for the new generation ---
            total_bursts_activated_gen = 0
            total_burst_energy_spent_gen = 0

            print(f"--- Generation {current_generation} started. Population: {len(creatures)}. Top Breeder Food: {survivors[0].food_eaten_individual if survivors else 'N/A'} ---")


        # Spawn new food if below max limit
        if len(food_items) < const.MAX_FOOD_COUNT:
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
                # Before removing, add its lifetime burst stats to the generation totals
                # Removed 'global' keywords as these vars are already in global scope
                total_bursts_activated_gen += creature.bursts_activated_individual
                total_burst_energy_spent_gen += creature.energy_spent_bursting_individual
                creatures.remove(creature)


    # --- Drawing/Rendering ---
    screen.fill(const.BACKGROUND_COLOR)

    for food in food_items:
        food.draw(screen)

    for creature in creatures:
        creature.draw(screen)

    # --- UI/Info Panel ---
    population_text = font.render(f"Population: {len(creatures)}", True, const.WHITE)
    screen.blit(population_text, (10, 10))
    food_in_world_text = font.render(f"Food in World: {len(food_items)}", True, const.WHITE)
    screen.blit(food_in_world_text, (10, 40))
    total_eaten_text = font.render(f"Food Eaten (Total): {food_eaten_count}", True, const.WHITE)
    screen.blit(total_eaten_text, (10, 70))
    gen_eaten_text = font.render(f"Food Eaten (Gen): {food_eaten_this_generation}/{const.FOOD_LIMIT_PER_GENERATION}", True, const.WHITE)
    screen.blit(gen_eaten_text, (10, 100))
    gen_text = font.render(f"Generation: {current_generation}", True, const.WHITE)
    screen.blit(gen_text, (10, 130))
    bursts_text = font.render(f"Gen Bursts: {total_bursts_activated_gen}", True, const.WHITE) # NEW UI LINE
    screen.blit(bursts_text, (10, 160)) # Adjusted position
    burst_energy_text = font.render(f"Gen Burst Energy: {total_burst_energy_spent_gen:.0f}", True, const.WHITE) # NEW UI LINE
    screen.blit(burst_energy_text, (10, 190)) # Adjusted position


    # --- Update the Display ---
    pygame.display.flip()

    # --- Control Frame Rate ---
    clock.tick(const.FPS)

# --- 5. Quit Pygame ---
pygame.quit()
if log_file:
    log_file.close()
sys.exit()