# constants.py

# --- Window and Display Settings ---
WIDTH, HEIGHT = 1000, 700
FPS = 60

# --- Color Definitions ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (20, 20, 30) # Dark background
FOOD_COLOR = (100, 255, 100) # Green food
OBSTACLE_COLOR = (100, 100, 150) # Dark gray/blue for obstacles

# --- Creature Properties ---
INITIAL_CREATURE_COUNT = 50 # Initial number of creatures
CREATURE_RADIUS = 5 # <--- ADD THIS LINE (or ensure it's present)
CREATURE_BASE_SPEED = 2.0 # Base movement speed
CREATURE_ENERGY_DECAY = 0.05
CREATURE_MAX_ENERGY = 100 # Maximum energy for a creature

# --- Food Properties ---
FOOD_RADIUS = 3
FOOD_ENERGY_GAIN = 40
MAX_FOOD_COUNT = 250 # Maximum food items in world

# --- Mutation Settings ---
MUTATION_CHANCE = 0.02 # Chance for a gene to mutate (0-1)
NN_MUTATION_AMOUNT = 0.1 # Amount NN weights/biases can change during mutation
COLOR_MUTATION_AMOUNT = 30 # Amount creature color can change during mutation

# --- Speed Burst Constants ---
BURST_ENERGY_COST = 10.0
BURST_DURATION_FRAMES = 60 # How many frames the burst lasts
BURST_NN_THRESHOLD = 0.5  # NN output must exceed this to trigger burst (tanh output)
BURST_FITNESS_BONUS = 5.0 # Bonus to fitness for each burst activated
BURST_SPEED_MULTIPLIER = 1.8 # e.g., 1.8x base speed

# --- Neural Network Architecture Constants ---
NN_INPUT_NODES = 8 # Energy, Food Distance, Food Angle, Neighbor Distance, Neighbor Speed, Neighbor Angle, Obstacle Distance, Obstacle Angle
NN_HIDDEN_NODES = 4 # Number of nodes in the neural network's hidden layer
NN_OUTPUT_NODES = 2 # 1 for steering, 1 for burst
MAX_TURN_ANGLE = 7 # Max degrees a creature can turn per frame

# --- Generation Control ---
GENERATION_LENGTH_FRAMES = 500 # Frames per generation
SELECTION_PERCENTAGE = 0.3 # Top percentage of creatures to breed (0-1)
FOOD_LIMIT_PER_GENERATION = 50 # Food items consumed to end generation early

# --- Obstacle Settings ---
OBSTACLE_PENALTY = 2       # Energy lost per collision with an obstacle
OBSTACLE_SIZE = 25 # Size of square obstacles (if you use fixed size obstacles)
NUM_OBSTACLES = 3 # Number of obstacles to generate

# --- Breeding and Fitness Metrics ---
COLLISION_PENALTY_BREEDING = 1.0 # Points subtracted from fitness for each collision

# --- Spawn Safety ---
MIN_SPAWN_DISTANCE_FROM_OBSTACLE = 25 # Minimum distance from obstacle edges for spawning

# --- Logging Configuration ---
LOG_DIRECTORY = "simulation_logs"
LIVE_LOG_FILE_NAME = "evolution_live_log.csv"