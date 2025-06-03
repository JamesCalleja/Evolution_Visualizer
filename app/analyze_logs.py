import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from constants import (log_filepath)

# --- Visualization Script ---
def analyze_simulation_logs(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Log file not found at {filepath}")
        print("Please run the simulation first to generate data.")
        return

    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        print("Error: Log file is empty. Run the simulation for some time.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    print("\n--- Data Head ---")
    print(df.head())

    print("\n--- Data Info ---")
    df.info()

    # Convert relevant columns to numeric, coercing errors
    numeric_cols = [
        "Food_Eaten_Gen",
        "Top_Creature_Food_Eaten",
        "Num_Survivors",
        "Population_Size_Start_Gen",
        "Avg_Survivor_Energy",
        "Frames_This_Gen",
        "Total_Bursts_Activated_Gen",  # NEW
        "Total_Burst_Energy_Spent_Gen" # NEW
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where critical numeric columns are NaN after coercion
    # Ensure new burst columns are not critical for dropping unless desired
    df.dropna(subset=["Food_Eaten_Gen", "Num_Survivors", "Top_Creature_Food_Eaten"], inplace=True)

    if df.empty:
        print("No valid data rows to plot after cleaning.")
        return

    print(f"\nAnalyzing {len(df)} generations.")

    # --- Plotting ---
    sns.set_style("whitegrid")
    # Adjusted figure size and subplot grid for 8 plots (4 rows, 2 columns)
    plt.figure(figsize=(16, 18)) # Increased height to accommodate more plots

    # Plot 1: Food Eaten per Generation
    plt.subplot(4, 2, 1) # Changed to 4 rows, 2 columns
    sns.lineplot(x="Generation", y="Food_Eaten_Gen", data=df)
    plt.title("Food Eaten Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Food Eaten")

    # Plot 2: Top Creature's Food Eaten
    plt.subplot(4, 2, 2) # Changed to 4 rows, 2 columns
    sns.lineplot(x="Generation", y="Top_Creature_Food_Eaten", data=df, color='orange')
    plt.title("Top Creature's Food Eaten Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Food Eaten by Best Individual")

    # Plot 3: Number of Survivors
    plt.subplot(4, 2, 3) # Changed to 4 rows, 2 columns
    sns.lineplot(x="Generation", y="Num_Survivors", data=df, color='green')
    plt.title("Number of Survivors Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Count")

    # Plot 4: Population Size at Start of Generation
    plt.subplot(4, 2, 4) # Changed to 4 rows, 2 columns
    sns.lineplot(x="Generation", y="Population_Size_Start_Gen", data=df, color='red')
    plt.title("Population Size at Start of Generation")
    plt.xlabel("Generation")
    plt.ylabel("Population Count")

    # Plot 5: Average Survivor Energy
    plt.subplot(4, 2, 5) # Changed to 4 rows, 2 columns
    sns.lineplot(x="Generation", y="Avg_Survivor_Energy", data=df, color='purple')
    plt.title("Average Energy of Survivors Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Average Energy")

    # Plot 6: Frames This Generation (Duration)
    plt.subplot(4, 2, 6) # Changed to 4 rows, 2 columns
    sns.lineplot(x="Generation", y="Frames_This_Gen", data=df, color='brown')
    plt.title("Frames per Generation (Duration)")
    plt.xlabel("Generation")
    plt.ylabel("Frames")

    # NEW Plot 7: Total Bursts Activated Per Generation
    plt.subplot(4, 2, 7) # New subplot
    sns.lineplot(x="Generation", y="Total_Bursts_Activated_Gen", data=df, color='blue')
    plt.title("Total Bursts Activated Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Number of Bursts")

    # NEW Plot 8: Total Burst Energy Spent Per Generation
    plt.subplot(4, 2, 8) # New subplot
    sns.lineplot(x="Generation", y="Total_Burst_Energy_Spent_Gen", data=df, color='cyan')
    plt.title("Total Energy Spent on Bursts Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Energy Spent")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjusted layout to make space for suptitle
    plt.suptitle("Evolution Simulation Metrics Over Generations (Including Speed Burst)", y=1.00, fontsize=16) # Updated title
    plt.show()

    # --- Optional: Correlation Heatmap ---
    print("\n--- Correlation Matrix ---")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix)

    plt.figure(figsize=(10, 8)) # Adjusted size for potentially more columns in heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of Simulation Metrics")
    plt.show()

if __name__ == "__main__":
    analyze_simulation_logs(log_filepath)