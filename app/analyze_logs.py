# analyze_logs.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
LOG_DIRECTORY = "simulation_logs"
LIVE_LOG_FILE_NAME = "evolution_live_log.csv"
log_filepath = os.path.join(LOG_DIRECTORY, LIVE_LOG_FILE_NAME)

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
        "Top_Creature_Food_Eaten_Individual",
        "Top_Creature_Energy_Individual",
        "Top_Creature_Collisions_Individual",
        "Top_Creature_Bursts_Activated_Individual", # NEW
        "Top_Creature_Combined_Fitness",
        "Num_Survivors",
        "Population_Size_Start_Gen",
        "Avg_Survivor_Energy",
        "Frames_This_Gen",
        "Total_Bursts_Activated_Gen",
        "Total_Burst_Energy_Spent_Gen",
        "Total_Collisions_Gen"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where critical numeric columns are NaN after coercion
    df.dropna(subset=["Food_Eaten_Gen", "Num_Survivors", "Top_Creature_Food_Eaten_Individual"], inplace=True)

    if df.empty:
        print("No valid data rows to plot after cleaning.")
        return

    print(f"\nAnalyzing {len(df)} generations.")

    # --- Plotting ---
    # Adjusted figure size and subplot grid for 12 plots (6 rows, 2 columns)
    plt.figure(figsize=(16, 30)) # Increased height to accommodate 6 rows of plots and 12 plots

    # Plot 1: Food Eaten per Generation
    plt.subplot(6, 2, 1) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Food_Eaten_Gen", data=df)
    plt.title("Food Eaten Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Food Eaten")

    # Plot 2: Top Creature's Food Eaten
    plt.subplot(6, 2, 2) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Top_Creature_Food_Eaten_Individual", data=df, color='orange')
    plt.title("Top Creature's Food Eaten Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Food Eaten by Best Individual")

    # Plot 3: Number of Survivors
    plt.subplot(6, 2, 3) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Num_Survivors", data=df, color='green')
    plt.title("Number of Survivors Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Count")

    # Plot 4: Population Size at Start of Generation
    plt.subplot(6, 2, 4) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Population_Size_Start_Gen", data=df, color='red')
    plt.title("Population Size at Start of Generation")
    plt.xlabel("Generation")
    plt.ylabel("Population Size")

    # Plot 5: Top Creature's Energy (Individual)
    plt.subplot(6, 2, 5) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Top_Creature_Energy_Individual", data=df, color='purple')
    plt.title("Top Creature's Energy Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Energy")

    # Plot 6: Average Survivor Energy
    plt.subplot(6, 2, 6) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Avg_Survivor_Energy", data=df, color='brown')
    plt.title("Average Survivor Energy Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Average Energy")

    # Plot 7: Top Creature's Combined Fitness
    plt.subplot(6, 2, 7) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Top_Creature_Combined_Fitness", data=df, color='blue')
    plt.title("Top Creature's Combined Fitness Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")

    # Plot 8: Frames Per Generation
    plt.subplot(6, 2, 8) # Changed to 6 rows, 2 columns
    sns.lineplot(x="Generation", y="Frames_This_Gen", data=df, color='cyan')
    plt.title("Frames Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Frames")

    # Plot 9: Total Bursts Activated Per Generation
    plt.subplot(6, 2, 9)
    sns.lineplot(x="Generation", y="Total_Bursts_Activated_Gen", data=df, color='magenta')
    plt.title("Total Bursts Activated Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Total Bursts")

    # Plot 10: Total Burst Energy Spent Per Generation
    plt.subplot(6, 2, 10)
    sns.lineplot(x="Generation", y="Total_Burst_Energy_Spent_Gen", data=df, color='lime')
    plt.title("Total Burst Energy Spent Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Energy Spent on Bursts")

    # Plot 11: Top Creature's Collisions (Individual)
    plt.subplot(6, 2, 11)
    sns.lineplot(x="Generation", y="Top_Creature_Collisions_Individual", data=df, color='darkred')
    plt.title("Top Creature's Collisions Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Collisions by Best Individual")
    
    # Plot 12: Total Collisions Per Generation
    plt.subplot(6, 2, 12) # Corrected subplot index
    sns.lineplot(x="Generation", y="Total_Collisions_Gen", data=df, color='gray') # Changed color for variety
    plt.title("Total Collisions Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Total Collisions")


    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjusted layout to make space for suptitle
    plt.suptitle("Evolution Simulation Metrics Over Generations (Including Speed Burst and Obstacles)", y=1.00, fontsize=16) # Updated title
    plt.show()

    # --- Optional: Correlation Heatmap ---
    print("\n--- Correlation Matrix ---")
    
    numeric_cols_for_corr = [
        "Food_Eaten_Gen",
        "Top_Creature_Food_Eaten_Individual",
        "Top_Creature_Energy_Individual",
        "Top_Creature_Collisions_Individual",
        "Top_Creature_Bursts_Activated_Individual", # NEW
        "Top_Creature_Combined_Fitness",
        "Num_Survivors",
        "Population_Size_Start_Gen",
        "Avg_Survivor_Energy",
        "Frames_This_Gen",
        "Total_Bursts_Activated_Gen",
        "Total_Burst_Energy_Spent_Gen",
        "Total_Collisions_Gen",
        "top_creature_bursts_individual"
    ]

    correlation_matrix = df[numeric_cols_for_corr].corr()
    print(correlation_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Simulation Metrics")
    plt.show()


if __name__ == "__main__":
    analyze_simulation_logs(log_filepath)