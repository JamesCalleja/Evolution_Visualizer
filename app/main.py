import pygame
import csv
import os
import sys
import random
import math

# Import components from your new modules
import constants as const
from creatures import Creature, Food, Obstacle 

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
obstacles = []

# --- NEW: Global counters for burst data per generation ---
total_bursts_activated_gen = 0
total_burst_energy_spent_gen = 0


def create_initial_population(count, obstacles):
    new_population = []
    for _ in range(count):
        # Pass obstacles_ref to Creature constructor
        new_population.append(Creature(obstacles_ref=obstacles))
    return new_population

def initialize_obstacles(num_obstacles, width, height, obstacle_size, min_spawn_distance, creatures=None, food_items=None):
    """
    Initializes a list of non-overlapping obstacles, ensuring they don't
    spawn too close to creatures or food if provided.
    """
    new_obstacles = []
    for _ in range(num_obstacles):
        while True:
            obs_x = random.randint(0, width - obstacle_size)
            obs_y = random.randint(0, height - obstacle_size)
            new_obs = Obstacle(obs_x, obs_y, obstacle_size, obstacle_size)

            # Check for overlap with existing obstacles
            overlap = False
            for existing_obs in new_obstacles:
                if (obs_x < existing_obs.x + existing_obs.width and
                    obs_x + obstacle_size > existing_obs.x and
                    obs_y < existing_obs.y + existing_obs.height and
                    obs_y + obstacle_size > existing_obs.y):
                    overlap = True
                    break
            if overlap:
                continue

            # Check for overlap with creatures (if provided)
            if creatures:
                for creature in creatures:
                    # Simple circle-rect collision for creature-obstacle during spawn
                    closest_x = max(new_obs.x, min(creature.x, new_obs.x + new_obs.width))
                    closest_y = max(new_obs.y, min(creature.y, new_obs.y + new_obs.height))
                    distance = math.hypot(creature.x - closest_x, creature.y - closest_y)
                    if distance < creature.radius + min_spawn_distance:
                        overlap = True
                        break
            if overlap:
                continue

            # Check for overlap with food (if provided)
            if food_items:
                for food in food_items:
                    closest_x = max(new_obs.x, min(food.x, new_obs.x + new_obs.width))
                    closest_y = max(new_obs.y, min(food.y, new_obs.y + new_obs.height))
                    distance = math.hypot(food.x - closest_x, food.y - closest_y)
                    if distance < food.radius + min_spawn_distance:
                        overlap = True
                        break
            if overlap:
                continue

            new_obstacles.append(new_obs)
            break
    return new_obstacles

# Initialize obstacles
obstacles = initialize_obstacles(const.NUM_OBSTACLES, const.WIDTH, const.HEIGHT, const.OBSTACLE_SIZE, const.MIN_SPAWN_DISTANCE_FROM_OBSTACLE)

# Pass the obstacles list when creating initial population
creatures = create_initial_population(const.INITIAL_CREATURE_COUNT, obstacles)

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
                        "Generation", "Food_Eaten_Gen", "Top_Creature_Food_Eaten",
                        "Num_Survivors", "Population_Size_Start_Gen", "Avg_Survivor_Energy",
                        "Total_Bursts_Activated_Gen",
                        "Total_Burst_Energy_Spent_Gen",
                        "Avg_Fitness",
                        "Avg_Turning_Rate", # NEW: Average Turning Rate
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
                obstacles = initialize_obstacles(const.NUM_OBSTACLES, const.WIDTH, const.HEIGHT, const.OBSTACLE_SIZE, const.MIN_SPAWN_DISTANCE_FROM_OBSTACLE)
                creatures = create_initial_population(const.INITIAL_CREATURE_COUNT, obstacles)
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
                        current_generation,
                        food_eaten_this_generation,
                        top_creature_food_eaten,
                        len(survivors_for_log_list),
                        len(creatures),
                        f"{avg_survivor_energy:.2f}",
                        total_bursts_activated_gen,
                        total_burst_energy_spent_gen,
                        f"{avg_survivor_fitness:.2f}",
                        f"{0.0:.2f}",  # avg_turning_rate placeholder
                        frame_count_this_generation
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

                top_creature_food_eaten = living_creatures_for_log[0].food_eaten_individual if living_creatures_for_log else 0
                
                # --- MOVE THESE LINES UP ---
                num_survivors_for_log = max(1, int(len(living_creatures_for_log) * const.SELECTION_PERCENTAGE))
                survivors_for_log_list = []
                survivors_for_log_list = living_creatures_for_log[:num_survivors_for_log]
                # --- END OF MOVE ---

                # Calculate average fitness (this line caused the error)
                avg_survivor_fitness = sum(c.calculate_fitness() for c in survivors_for_log_list) / len(survivors_for_log_list) if survivors_for_log_list else 0

                avg_survivor_energy = sum(c.energy for c in survivors_for_log_list) / len(survivors_for_log_list) if survivors_for_log_list else 0

                # Calculate average turning rate (if attribute exists)
                if survivors_for_log_list and hasattr(survivors_for_log_list[0], "turning_rate"):
                    avg_turning_rate = sum(getattr(c, "turning_rate", 0) for c in survivors_for_log_list) / len(survivors_for_log_list)
                else:
                    avg_turning_rate = 0.0

                log_writer.writerow([
                    current_generation,
                    food_eaten_this_generation,
                    top_creature_food_eaten,
                    len(survivors_for_log_list),
                    len(creatures),
                    f"{avg_survivor_energy:.2f}",
                    total_bursts_activated_gen,
                    total_burst_energy_spent_gen,
                    f"{avg_survivor_fitness:.2f}",
                    f"{avg_turning_rate:.2f}",
                    frame_count_this_generation
                ])
                log_file.flush()

            current_generation += 1
            frame_count_this_generation = 0

            # 1. Selection
            living_creatures_for_selection = [c for c in creatures if not c.is_dying]
            living_creatures_for_selection.sort(key=lambda c: c.calculate_fitness(), reverse=True) 

            num_survivors = max(1, int(len(living_creatures_for_selection) * const.SELECTION_PERCENTAGE))
            survivors = living_creatures_for_selection[:num_survivors]

            new_generation_creatures = []
            if not survivors:
                # Fallback mechanism: Select based on food eaten if no fitness-based survivors
                print(f"Gen {current_generation-1}: No fitness-based survivors. Selecting creatures based on food eaten for breeding.")
                
                # Consider all creatures from the just-finished generation (even those dying but with stats)
                # The 'creatures' list here still holds the creatures from the just-finished generation.
                all_creatures_past_gen_sorted_by_food = sorted(creatures, key=lambda c: c.calculate_fitness(), reverse=True)
                
                fallback_num_breeders = max(1, int(len(all_creatures_past_gen_sorted_by_food) * const.SELECTION_PERCENTAGE))
                fallback_breeders = all_creatures_past_gen_sorted_by_food[:fallback_num_breeders]
                
                # Use these fallback breeders to create the new generation
                for _ in range(const.INITIAL_CREATURE_COUNT):
                    # Ensure there are breeders to choose from
                    if not fallback_breeders:
                        print(f"Gen {current_generation-1}: No fallback breeders found. Breaking reproduction loop.")
                        break
                    parent = random.choice(fallback_breeders)
                    offspring = parent.reproduce()
                    if offspring:
                        new_generation_creatures.append(offspring)
                        if len(new_generation_creatures) >= const.INITIAL_CREATURE_COUNT:
                            break
                
                # If even with fallback, no offspring are produced (e.g., fallback_breeders is empty or reproduce fails)
                if not new_generation_creatures:
                    print(f"Gen {current_generation-1}: No creatures selected even with food-based fallback. Re-initializing population.")
                    new_generation_creatures = create_initial_population(const.INITIAL_CREATURE_COUNT, obstacles)
            else:
                # Fill remaining spots with offspring from fitness-based survivors
                for _ in range(const.INITIAL_CREATURE_COUNT):
                    parent = random.choice(survivors)
                    offspring = parent.reproduce()
                    if offspring:
                        new_generation_creatures.append(offspring)
                        if len(new_generation_creatures) >= const.INITIAL_CREATURE_COUNT:
                            break

            creatures = []
            creatures.extend(new_generation_creatures)
            food_items = [] 
            
            obstacles = []
            obstacles = initialize_obstacles(
                const.NUM_OBSTACLES,
                const.WIDTH,
                const.HEIGHT,
                const.OBSTACLE_SIZE,
                const.MIN_SPAWN_DISTANCE_FROM_OBSTACLE,
                creatures=creatures,
                food_items=food_items)
            
            for creature in creatures:
                creature.obstacles_ref = obstacles

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
            creature.move(food_items, obstacles )

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
        
    for obs in obstacles:
        obs.draw(screen)

    for creature in creatures:
            if not creature.is_dying:
                creature.draw(screen)
                creature.move(food_items, obstacles)

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
