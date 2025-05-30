import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import time # Import time for potential brief delay

# Get default values from evolution_visualizer.py for initial GUI population
DEFAULT_PARAMS = {
    'initial_creature_count': 50,
    'creature_energy_decay': 0.05,
    'food_energy_gain': 40.0,
    'max_food_count': 500,
    'mutation_chance': 0.02,
    'nn_mutation_amount': 0.1,
    'color_mutation_amount': 30,
    'generation_length_frames': 5000,
    'selection_percentage': 0.3,
    'creature_max_energy': 100.0,
    'food_limit_per_generation': 500,
    'log_enabled': True,
    'width': 1000,
    'height': 700,
    'fps': 60
}

class EvolutionSimulatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Evolution Simulator Control")
        self.geometry("500x750")
        self.simulation_process = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.param_vars = {}
        row = 0

        # Create input fields for each parameter
        self.add_separator(main_frame, "General Settings", row); row += 1
        self.add_param_entry(main_frame, "Window Width:", 'width', int, row); row += 1
        self.add_param_entry(main_frame, "Window Height:", 'height', int, row); row += 1
        self.add_param_entry(main_frame, "FPS:", 'fps', int, row); row += 1

        self.add_separator(main_frame, "Creature & Food Dynamics", row); row += 1
        self.add_param_entry(main_frame, "Initial Creatures:", 'initial_creature_count', int, row); row += 1
        self.add_param_entry(main_frame, "Energy Decay:", 'creature_energy_decay', float, row); row += 1
        self.add_param_entry(main_frame, "Food Energy Gain:", 'food_energy_gain', float, row); row += 1
        self.add_param_entry(main_frame, "Max Food Items:", 'max_food_count', int, row); row += 1
        self.add_param_entry(main_frame, "Max Creature Energy:", 'creature_max_energy', float, row); row += 1

        self.add_separator(main_frame, "Evolution & Mutation", row); row += 1
        self.add_param_entry(main_frame, "Mutation Chance (0-1):", 'mutation_chance', float, row); row += 1
        self.add_param_entry(main_frame, "NN Mutate Amount:", 'nn_mutation_amount', float, row); row += 1
        self.add_param_entry(main_frame, "Color Mutate Amount:", 'color_mutation_amount', int, row); row += 1
        self.add_param_entry(main_frame, "Selection % (0-1):", 'selection_percentage', float, row); row += 1

        self.add_separator(main_frame, "Generation Control", row); row += 1
        self.add_param_entry(main_frame, "Gen Length (Frames):", 'generation_length_frames', int, row); row += 1
        self.add_param_entry(main_frame, "Food Limit Per Gen:", 'food_limit_per_generation', int, row); row += 1

        self.add_separator(main_frame, "Logging", row); row += 1
        self.param_vars['log_enabled'] = tk.BooleanVar(value=DEFAULT_PARAMS['log_enabled'])
        log_checkbox = ttk.Checkbutton(main_frame, text="Enable Logging", variable=self.param_vars['log_enabled'])
        log_checkbox.grid(row=row, column=0, columnspan=2, pady=5, sticky="w")
        row += 1

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)

        self.run_button = ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop Simulation", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

    def add_separator(self, parent, text, row):
        sep = ttk.Separator(parent, orient='horizontal')
        sep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 5))
        label = ttk.Label(parent, text=text, font=("Arial", 10, "bold"))
        label.grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=5)
        self.grid_rowconfigure(row+1, weight=1)

    def add_param_entry(self, parent, label_text, param_key, var_type, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", pady=2, padx=5)

        if var_type == int:
            var = tk.IntVar(value=DEFAULT_PARAMS[param_key])
            validate_cmd = self.register(self.validate_int_input)
            entry = ttk.Entry(parent, textvariable=var, validate="key", validatecommand=(validate_cmd, '%P'))
        elif var_type == float:
            var = tk.DoubleVar(value=DEFAULT_PARAMS[param_key])
            validate_cmd = self.register(self.validate_float_input)
            entry = ttk.Entry(parent, textvariable=var, validate="key", validatecommand=(validate_cmd, '%P'))
        else:
            var = tk.StringVar(value=str(DEFAULT_PARAMS[param_key]))
            entry = ttk.Entry(parent, textvariable=var)

        entry.grid(row=row, column=1, sticky="ew", pady=2, padx=5)
        self.param_vars[param_key] = var
        parent.grid_columnconfigure(1, weight=1)

    def validate_int_input(self, new_value):
        if new_value == "": return True
        try:
            int(new_value)
            return True
        except ValueError:
            return False

    def validate_float_input(self, new_value):
        if new_value == "": return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False

    def run_simulation(self):
        if self.simulation_process and self.simulation_process.poll() is None:
            messagebox.showinfo("Simulator Running", "Simulation is already running. Please stop it first.")
            return

        try:
            params = []
            for key, var in self.param_vars.items():
                if key == 'log_enabled':
                    params.append(f'--{key}')
                    params.append(str(int(var.get())))
                else:
                    params.append(f'--{key}')
                    params.append(str(var.get()))

            python_exe = sys.executable
            if python_exe.endswith("pythonw.exe"):
                python_exe = python_exe.replace("pythonw.exe", "python.exe")
                if not os.path.exists(python_exe):
                    messagebox.showwarning("Warning", f"python.exe not found at {python_exe}. Trying default sys.executable.")
                    python_exe = sys.executable

            command = [python_exe, 'evolution_visualizer.py'] + params
            print(f"Executing simulation command: {' '.join(command)}")
            print("--- Simulation Output (if any) ---")

            self.simulation_process = subprocess.Popen(
                command,
                # stdout=subprocess.PIPE, # Keep these commented for direct console output
                # stderr=subprocess.PIPE,
                # text=True
            )

            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
            print(f"Error starting simulation: {e}")

    def stop_simulation(self):
        if self.simulation_process:
            if self.simulation_process.poll() is None: # Check if process is still running
                try:
                    self.simulation_process.terminate() # Request graceful termination
                    self.simulation_process.wait(timeout=5) # Wait for it to terminate
                    print("Simulation process terminated.")
                except subprocess.TimeoutExpired:
                    print("Simulation process did not terminate gracefully, forcing kill.")
                    self.simulation_process.kill() # Force kill if terminate fails
                except Exception as e:
                    print(f"Error terminating process: {e}")
            else:
                print("Simulation process was already stopped.")

            # --- NEW: Trigger analyze_logs.py ---
            try:
                # Ensure the simulation process has definitely finished writing its log
                time.sleep(0.5) # Give it a little buffer time, in case of file flushing

                python_exe = sys.executable
                if python_exe.endswith("pythonw.exe"):
                    python_exe = python_exe.replace("pythonw.exe", "python.exe")
                    if not os.path.exists(python_exe):
                        print("Warning: python.exe not found for analysis script. Using default sys.executable.")
                        python_exe = sys.executable

                analysis_command = [python_exe, 'analyze_logs.py']
                print(f"\nExecuting analysis command: {' '.join(analysis_command)}")
                print("--- Analysis Output ---")
                
                # Run analysis script. We'll use subprocess.run for this as we want it to complete
                # and ideally show its output in the same console.
                subprocess.run(analysis_command, check=True) # check=True will raise an error if analysis script fails
                print("--- Analysis Complete ---")

            except FileNotFoundError:
                messagebox.showerror("Error", "analyze_logs.py not found. Make sure it's in the same directory.")
                print("Error: analyze_logs.py not found.")
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Analysis Error", f"analyze_logs.py failed with error:\n{e}")
                print(f"Analysis script failed: {e}")
            except Exception as e:
                messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis: {e}")
                print(f"Error launching analysis script: {e}")


            self.simulation_process = None # Clear the process reference
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        else:
            messagebox.showinfo("No Simulation", "No simulation is currently running.")

    def on_closing(self):
        if self.simulation_process and self.simulation_process.poll() is None:
            if messagebox.askyesno("Quit", "A simulation is running. Do you want to stop it and quit?"):
                self.stop_simulation()
                self.destroy()
            else:
                pass
        else:
            self.destroy()


if __name__ == "__main__":
    app = EvolutionSimulatorGUI()
    app.mainloop()