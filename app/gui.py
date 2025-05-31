import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import time # Import time for potential brief delay

# Get default values from main.py for initial GUI population
DEFAULT_PARAMS = {
    'initial_creature_count': 50,
    'creature_energy_decay': 0.05,
    'food_energy_gain': 15,
    'max_food_count': 50,
    'mutation_chance': 0.06,
    'nn_mutation_amount': 0.1,
    'color_mutation_amount': 30,
    'generation_length_frames': 500,
    'selection_percentage': 0.3,
    'creature_max_energy': 35,
    'food_limit_per_generation': 250,
    'log_enabled': True,
    'width': 1000,
    'height': 700,
    'fps': 60,
    'nn_hidden_nodes': 4 # NEW: Default value for NN Hidden Nodes
}

class EvolutionSimulatorGUI(tk.Tk):
    """
    A Tkinter-based GUI for controlling the Evolution Simulator.
    Allows users to adjust simulation parameters and start/stop the simulation.
    """
    def __init__(self):
        """Initializes the GUI application."""
        super().__init__()
        self.title("Evolution Simulator GUI")
        self.geometry("600x850") # Increased height to accommodate more controls

        self.simulation_process = None # To hold the subprocess reference

        self.create_widgets()
        self.load_default_params()
        
        # Set up a protocol for handling window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Creates and arranges the GUI widgets."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Use a canvas with a scrollbar for many parameters
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


        self.param_entries = {}
        row = 0

        # Simulation Parameters Section
        ttk.Label(scrollable_frame, text="Simulation Parameters", font=("Arial", 14, "bold")).grid(row=row, column=0, columnspan=2, pady=(10, 5), sticky="w")
        row += 1

        params_order = [
            ('initial_creature_count', 'Initial Creatures:', int),
            ('creature_energy_decay', 'Energy Decay Rate:', float),
            ('food_energy_gain', 'Food Energy Gain:', float),
            ('max_food_count', 'Max Food Items:', int),
            ('creature_max_energy', 'Creature Max Energy:', float),
            ('mutation_chance', 'Mutation Chance (0-1):', float),
            ('nn_mutation_amount', 'NN Mutation Amount:', float),
            ('color_mutation_amount', 'Color Mutation Amount:', int),
            ('nn_hidden_nodes', 'NN Hidden Nodes:', int), # Added NN Hidden Nodes
            ('selection_percentage', 'Selection % (0-1):', float),
            ('generation_length_frames', 'Generation Length (Frames):', int),
            ('food_limit_per_generation', 'Food Limit per Gen:', int),
            ('width', 'Window Width:', int),
            ('height', 'Window Height:', int),
            ('fps', 'FPS:', int),
        ]

        for key, label_text, type_cast in params_order:
            ttk.Label(scrollable_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(scrollable_frame)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.param_entries[key] = (entry, type_cast)
            row += 1

        # Checkbox for log_enabled
        self.log_enabled_var = tk.BooleanVar()
        log_checkbox = ttk.Checkbutton(scrollable_frame, text="Enable Logging", variable=self.log_enabled_var)
        log_checkbox.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        row += 1

        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.run_button = ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Simulation", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.analyze_button = ttk.Button(button_frame, text="Analyze Logs", command=self.analyze_logs)
        self.analyze_button.grid(row=0, column=2, padx=5)
        
        # Configure column weights for resizing
        scrollable_frame.grid_columnconfigure(1, weight=1)


    def load_default_params(self):
        """Loads default parameter values into the GUI entry fields."""
        for key, default_value in DEFAULT_PARAMS.items():
            if key in self.param_entries:
                entry, _ = self.param_entries[key]
                entry.delete(0, tk.END)
                entry.insert(0, str(default_value))
            elif key == 'log_enabled':
                self.log_enabled_var.set(default_value)

    def get_params(self):
        """Retrieves current parameter values from the GUI."""
        params = {}
        for key, (entry, type_cast) in self.param_entries.items():
            try:
                params[key] = type_cast(entry.get())
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {key}. Please enter a valid number.")
                return None
        params['log_enabled'] = self.log_enabled_var.get()
        return params

    def run_simulation(self):
        """
        Gathers parameters from the GUI and starts the simulation as a subprocess.
        """
        if self.simulation_process and self.simulation_process.poll() is None:
            messagebox.showinfo("Simulation Running", "A simulation is already running.")
            return

        params = self.get_params()
        if params is None: # Error in getting params
            return
        
        # Construct the command to run main.py
        # THIS IS THE CRUCIAL CHANGE: Ensure it points to 'main.py'
        script_path = os.path.join(os.path.dirname(__file__), 'main.py')
        
        command = [sys.executable, script_path]

        for key, value in params.items():
            # Handle boolean value for --log_enabled
            if key == 'log_enabled':
                command.append(f'--{key}')
                command.append('1' if value else '0')
            else:
                command.append(f'--{key}')
                command.append(str(value))

        print("Executing simulation command:", " ".join(command))
        print("--- Simulation Output (if any) ---")

        try:
            # Use Popen to run asynchronously, capturing output
            self.simulation_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start a non-blocking read for simulation output
            # self.after(100, self.check_simulation_output) # If you had a text widget for output

        except FileNotFoundError:
            messagebox.showerror("Error", "Python interpreter not found. Make sure Python is installed and in your PATH.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
            print(f"Error starting simulation: {e}")

    def stop_simulation(self):
        """
        Stops the running simulation subprocess.
        """
        if self.simulation_process and self.simulation_process.poll() is None:
            print("Terminating simulation process...")
            self.simulation_process.terminate()
            try:
                self.simulation_process.wait(timeout=5) # Wait for process to terminate
            except subprocess.TimeoutExpired:
                self.simulation_process.kill() # Force kill if not terminated
                print("Simulation process forcefully killed.")
            finally:
                self.simulation_process = None
                self.run_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                print("Simulation stopped.")
        else:
            messagebox.showinfo("No Simulation", "No simulation is currently running.")

    def analyze_logs(self):
        """
        Launches the analyze_logs.py script to visualize simulation data.
        """
        log_script_path = os.path.join(os.path.dirname(__file__), 'analyze_logs.py')
        
        if not os.path.exists(log_script_path):
            messagebox.showerror("Error", f"Analysis script not found at {log_script_path}")
            return

        # Pass the log file path to the analysis script
        log_directory = "simulation_logs" # Ensure this matches the constant in analyze_logs.py and main.py
        live_log_file_name = "evolution_live_log.csv" # Ensure this matches the constant
        full_log_filepath = os.path.join(log_directory, live_log_file_name)


        try:
            # Use Popen to run asynchronously so it doesn't block the GUI
            subprocess.Popen([sys.executable, log_script_path, full_log_filepath])
            print(f"Launching analysis script: {log_script_path} {full_log_filepath}")
        except FileNotFoundError:
            messagebox.showerror("Analysis Error", "Python interpreter not found. Cannot launch analysis script.")
        except subprocess.CalledProcessError as e:
                messagebox.showerror("Analysis Error", f"analyze_logs.py failed with error:\n{e}")
                print(f"Analysis script failed: {e}")
        except Exception as e:
                messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis: {e}")
                print(f"Error launching analysis script: {e}")

            # self.simulation_process = None # Clear the reference to the process
            # self.run_button.config(state=tk.NORMAL)
            # self.stop_button.config(state=tk.DISABLED)


    def on_closing(self):
        """
        Handles the window closing event, prompting the user if a simulation is running.
        """
        if self.simulation_process and self.simulation_process.poll() is None:
            # If simulation is still running, ask user to confirm stopping it
            if messagebox.askyesno("Quit", "A simulation is running. Do you want to stop it and quit?"):
                self.stop_simulation() # Stop the simulation
                self.destroy() # Close the GUI window
            else:
                pass # Do nothing, keep the window open
        else:
            self.destroy() # Close the GUI window directly if no simulation is running


if __name__ == "__main__":
    app = EvolutionSimulatorGUI()
    app.mainloop()
