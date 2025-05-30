# Evolution Simulator

This project simulates the basic principles of natural selection and evolution using simple "critters" (creatures) that learn to find food in a 2D environment. Creatures with neural networks that are more successful at finding food survive, reproduce, and pass on their "genetic" (neural network) traits, leading to observable evolutionary behaviors over generations.

---

## Features

* **Generational Evolution:** Creatures evolve over discrete generations, with selection based on fitness (food consumed).
* **Neural Network Brains:** Each creature possesses a small neural network that takes sensory input from the environment and outputs a steering force.
* **Basic Environmental Sensing:** Creatures can sense their own energy levels, the distance to the nearest food, and the angle to the nearest food.
* **Genetic Mutation:** Neural network weights/biases and creature colors undergo slight random mutations during reproduction, introducing variation for natural selection to act upon.
* **Real-time Visualization:** A Pygame window displays the creatures, food, and their movement.
* **Live Evolution Graph:** A separate Matplotlib window provides a real-time plot of key evolutionary metrics (e.g., top creature's food eaten, generation length) as the simulation progresses.
* **Detailed Logging:** Comprehensive data for each generation is saved to a CSV log file for later in-depth analysis.

---

## How It Works

1.  **Initialization:** A population of creatures with randomly initialized neural networks and colors is created. Food items are randomly scattered.
2.  **Simulation Loop (Per Generation):**
    * Creatures move around the environment, constantly spending energy.
    * They use their neural networks to process sensory data (energy, food distance, food angle) and decide how to steer.
    * If a creature collides with food, it consumes the food, gains energy, and its individual "food eaten" count increases.
    * Creatures lose energy over time. If their energy drops to zero, they "die" and are removed from the simulation.
    * The generation continues until either a set amount of food is consumed by the population or a maximum number of frames (time) has passed.
3.  **Selection & Reproduction:**
    * At the end of a generation, a specified percentage of the most successful creatures (those that ate the most food) are selected as "survivors."
    * These survivors become parents, reproducing to create the next generation. Offspring inherit the parents' neural network weights/biases and colors, with small mutations introduced.
    * The environment is reset with new food, and the cycle repeats.
4.  **Logging & Visualization:** Key statistics for each completed generation are recorded in a CSV file, and the real-time graph updates to show performance trends.

---

## Getting Started

To run this simulation, you'll need Python and a few libraries. It's **highly recommended** to use a virtual environment to manage dependencies.

### 1. Prerequisites

* Python 3.8+ (You appear to be using 3.13.3)

### 2. Setup (Recommended: Using a Virtual Environment)

Navigate to your project directory in PowerShell (or Command Prompt):

```powershell
cd C:\Code\EvolutionSimulator