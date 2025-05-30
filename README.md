# Evolution Visualizer

This project provides a Python-based simulation of artificial evolution, where simple creatures with neural networks learn to navigate an environment and consume food. It includes a graphical user interface (GUI) for adjusting simulation parameters, real-time visualization of the evolutionary process, and post-simulation analysis tools.

## Features

* **Evolutionary Simulation:** Creatures with neural networks (NNs) navigate a 2D environment, seek and consume food, and evolve over generations.
* **Genetic Algorithm:** Implements selection based on food eaten, reproduction with genetic inheritance, and mutation of NN weights/biases and creature colors.
* **Customizable Parameters:** Adjust key evolutionary and environmental metrics through a user-friendly GUI.
* **Real-time Visualization:** Watch the simulation unfold in a Pygame window, observing creature behavior and environmental dynamics.
* **Live Logging:** Records vital statistics (food eaten, survivor count, population size, etc.) for each generation into a CSV file.
* **Post-Simulation Analysis:** Automatically generates insightful plots and a correlation matrix of evolutionary trends after the simulation is stopped.
* **Interactive GUI:** Start, stop, and control simulation parameters from a dedicated Tkinter application.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.8+ installed on your system.
It's highly recommended to use a virtual environment to manage dependencies.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JamesCalleja/Evolution_Visualizer-.git](https://github.com/JamesCalleja/Evolution_Visualizer-.git)
    cd Evolution_Visualizer-
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source ./.venv/bin/activate
    ```
    (Note: Your directory listing showed `simvis` and `.venv`, if you prefer `simvis`, use `python -m venv simvis` and activate accordingly.)

3.  **Install the required Python packages:**
    ```bash
    pip install pygame pandas matplotlib seaborn
    ```

## How to Run the Simulator

The primary way to run the simulation and interact with its parameters is through the GUI.

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source ./.venv/bin/activate
    ```
2.  **Run the GUI application:**
    ```bash
    python gui.py
    ```
3.  **Adjust Parameters:** The GUI window will appear. You can modify various simulation metrics such as initial creature count, mutation rates, generation length, and more.
4.  **Start the Simulation:** Click the **"Run Simulation"** button. A new Pygame window will open, displaying the simulation. The GUI will remain responsive.
5.  **Stop the Simulation:** Click the **"Stop Simulation"** button on the GUI. This will close the Pygame window.

## Post-Simulation Analysis

Immediately after you click **"Stop Simulation"** in the GUI, the `analyze_logs.py` script will automatically execute. This script reads the `evolution_live_log.csv` file (located in the `simulation_logs/` directory) and generates a series of plots and a correlation matrix, providing insights into the evolutionary progress of your simulation run.

## Project Structure

* `evolution_visualizer.py`: The core simulation logic, including creature behavior, neural networks, genetic algorithm, and Pygame rendering. It accepts parameters via command-line arguments.
* `gui.py`: The Tkinter-based graphical user interface for controlling simulation parameters and launching/stopping the simulation. It orchestrates the running of `evolution_visualizer.py` as a subprocess.
* `analyze_logs.py`: A script that reads the simulation log data (`evolution_live_log.csv`) and generates visualizations (charts, correlation heatmap) using `pandas`, `matplotlib`, and `seaborn`.
* `simulation_logs/`: A directory created by `evolution_visualizer.py` to store the `evolution_live_log.csv` file, which records generational statistics.
* `README.md`: This file.
* `LICENSE`: Project license information.
* `.gitignore`: Specifies files and directories to be ignored by Git (e.g., virtual environment folders, `__pycache__`).

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.