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
        "Top_Creature_Food_Eaten",
        "Num_Survivors",
        "Population_Size_Start_Gen",
        "Avg_Survivor_Energy",
        "Frames_This_Gen"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where critical numeric columns are NaN after coercion
    df.dropna(subset=["Food_Eaten_Gen", "Num_Survivors", "Top_Creature_Food_Eaten"], inplace=True)

    if df.empty:
        print("No valid data rows to plot after cleaning.")
        return

    print(f"\nAnalyzing {len(df)} generations.")

    # --- Plotting ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))

    # Plot 1: Food Eaten per Generation
    plt.subplot(3, 2, 1)
    sns.lineplot(x="Generation", y="Food_Eaten_Gen", data=df)
    plt.title("Food Eaten Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Food Eaten")

    # Plot 2: Top Creature's Food Eaten
    plt.subplot(3, 2, 2)
    sns.lineplot(x="Generation", y="Top_Creature_Food_Eaten", data=df, color='orange')
    plt.title("Top Creature's Food Eaten Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Food Eaten by Best Individual")

    # Plot 3: Number of Survivors
    plt.subplot(3, 2, 3)
    sns.lineplot(x="Generation", y="Num_Survivors", data=df, color='green')
    plt.title("Number of Survivors Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Count")

    # Plot 4: Population Size at Start of Generation
    plt.subplot(3, 2, 4)
    sns.lineplot(x="Generation", y="Population_Size_Start_Gen", data=df, color='red')
    plt.title("Population Size at Start of Generation")
    plt.xlabel("Generation")
    plt.ylabel("Population Count")

    # Plot 5: Average Survivor Energy
    plt.subplot(3, 2, 5)
    sns.lineplot(x="Generation", y="Avg_Survivor_Energy", data=df, color='purple')
    plt.title("Average Energy of Survivors Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Average Energy")

    # Plot 6: Frames This Generation (Duration)
    plt.subplot(3, 2, 6)
    sns.lineplot(x="Generation", y="Frames_This_Gen", data=df, color='brown')
    plt.title("Frames per Generation (Duration)")
    plt.xlabel("Generation")
    plt.ylabel("Frames")


    plt.tight_layout()
    plt.suptitle("Evolution Simulation Metrics Over Generations", y=1.02, fontsize=16)
    plt.show()

    # --- Optional: Correlation Heatmap ---
    print("\n--- Correlation Matrix ---")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of Simulation Metrics")
    plt.show()

if __name__ == "__main__":
    analyze_simulation_logs(log_filepath)