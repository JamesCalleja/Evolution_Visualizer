import os
import argparse

# --- 1. Constants and Configuration ---
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

# --- NEW: Speed Burst Constants ---
BURST_ENERGY_COST = 10     # Energy consumed for one burst
BURST_SPEED_MULTIPLIER = 1.8 # How much faster they move (e.g., 1.8x base speed)
BURST_DURATION_FRAMES = 10 # How many frames the burst lasts
BURST_NN_THRESHOLD = 0.5   # NN output must exceed this to trigger burst (for tanh output, range -1 to 1)


# Neural Network Architecture Constants
NN_INPUT_NODES = 3  # Energy, Normalized Distance to Food, Normalized Angle to Food
NN_HIDDEN_NODES = 4
NN_OUTPUT_NODES = 2 # Changed from 1 to 2: 1 for steering, 1 for burst


# Generation Control Defaults
DEFAULT_GENERATION_LENGTH_FRAMES = 500
DEFAULT_SELECTION_PERCENTAGE = 0.3
DEFAULT_CREATURE_MAX_ENERGY = 100

# Generation End Metric Defaults
DEFAULT_FOOD_LIMIT_PER_GENERATION = 50

# Logging Configuration Defaults
DEFAULT_LOG_DIRECTORY = "../simulation_logs"
DEFAULT_LOG_ENABLED = True
LIVE_LOG_FILE_NAME = "evolution_live_log.csv"
log_filepath = os.path.join(DEFAULT_LOG_DIRECTORY, LIVE_LOG_FILE_NAME)


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
INITIAL_CREATURE_COUNT = args.initial_creature_count
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
LOG_ENABLED = bool(args.log_enabled)