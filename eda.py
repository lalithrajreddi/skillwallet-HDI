import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Starting Exploratory Data Analysis (EDA) with 10 visualization tasks...")
    local_processed_path = "hdi_processed.csv"
    plots_dir = os.path.join("static", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(local_processed_path):
        print(f"Error: Processed dataset '{local_processed_path}' not found! Run download_data.py first.")
        return False
        
    df = pd.read_csv(local_processed_path)
    
    # Task 1: Check unique country values and confirm no duplicates
    # "Unique Country Values: Use the unique() method on the Country column to list all unique country names, confirming there are no duplicate country entries."
    unique_countries = df['Country Name'].unique()
    print(f"Task 1: Unique Country names count: {len(unique_countries)}")
    print(f"Confirming no duplicates: Total rows = {len(df)}, Unique countries = {len(unique_countries)}")
    if len(df) == len(unique_countries):
        print("Verification Success: There are no duplicate country entries in the dataset.")
    else:
        print("Warning: There are duplicate country entries in the dataset.")

    # Slice the dataset to the first 20 rows (data1) to avoid overcrowding in plots
    data1 = df.head(20).copy()
    print(f"Using first {len(data1)} rows (data1) for plotting tasks to prevent overcrowding.")

    # Configure beautiful visualization aesthetics matching dark theme
    sns.set_theme(style="dark", rc={
        "axes.facecolor": "#161b2a",
        "figure.facecolor": "#0b0f19",
        "text.color": "#e2e8f0",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "axes.edgecolor": "#334155",
        "patch.edgecolor": "#161b2a"
    })
    
    tier_colors = {
        "Very High": "#06b6d4",
        "High": "#10b981",
        "Medium": "#f59e0b",
        "Low": "#ef4444"
    }
    
    # Task 2: Mean Years of Schooling vs HDI Score using a Strip Plot (Specifically required in Epic 3 Story 3)
    print("Task 2: Generating Mean Years of Schooling vs HDI Strip Plot...")
    plt.figure(figsize=(8, 5))
    sns.stripplot(data=data1, x="Mean yrs of schooling", y="HDI Score", hue="Development Tier",
                  palette=tier_colors, size=8, jitter=0.1, alpha=0.9, dodge=False)
    plt.title("Mean Years of Schooling vs HDI Score (First 20 Countries)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Mean Years of Schooling", fontsize=10)
    plt.ylabel("HDI Score", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "schooling_vs_hdi_strip.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # Task 3: Life Expectancy vs HDI Score (Scatter/Strip plot)
    print("Task 3: Generating Life Expectancy vs HDI Plot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data1, x="Life expectancy", y="HDI Score", hue="Development Tier",
                    palette=tier_colors, s=100, alpha=0.9, edgecolor="#0b0f19")
    plt.title("Life Expectancy vs HDI Score (First 20 Countries)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Life Expectancy (Years)", fontsize=10)
    plt.ylabel("HDI Score", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "life_expectancy_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # Task 4: Correlation Heatmap
    print("Task 4: Generating Correlation Heatmap...")
    plt.figure(figsize=(7, 5.5))
    corr_cols = ['Life expectancy', 'Expected yrs of schooling', 'Mean yrs of schooling', 'GNI per capita', 'HDI Score']
    corr_matrix = data1[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", 
                cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5, linecolor="#0b0f19")
    plt.title("Correlation Matrix of HDI Component Indicators", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    # Task 5: Expected Years of Schooling vs HDI Score Scatter Plot
    print("Task 5: Generating Expected Years of Schooling vs HDI Scatter Plot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data1, x="Expected yrs of schooling", y="HDI Score", hue="Development Tier",
                    palette=tier_colors, s=100, alpha=0.9, edgecolor="#0b0f19")
    plt.title("Expected Years of Schooling vs HDI Score", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Expected Years of Schooling", fontsize=10)
    plt.ylabel("HDI Score", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "expected_schooling_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    # Task 6: GNI per capita vs HDI Score (Log Scale) Scatter Plot
    print("Task 6: Generating GNI vs HDI Scatter Plot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data1, x="GNI per capita", y="HDI Score", hue="Development Tier",
                    palette=tier_colors, s=100, alpha=0.9, edgecolor="#0b0f19")
    plt.title("HDI Score vs Gross National Income (GNI) Per Capita", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("GNI Per Capita (PPP$)", fontsize=10)
    plt.ylabel("HDI Score", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gni_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    # Task 7: Distribution of HDI Score
    print("Task 7: Generating HDI Distribution Plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data1, x="HDI Score", kde=True, color="#8b5cf6", bins=10, fill=True, alpha=0.6)
    plt.title("Distribution of HDI Scores (First 20 Countries)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("HDI Score", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hdi_distribution.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    # Task 8: Distribution of Life Expectancy
    print("Task 8: Generating Life Expectancy Distribution Plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data1, x="Life expectancy", kde=True, color="#3b82f6", bins=10, fill=True, alpha=0.6)
    plt.title("Distribution of Life Expectancy", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Life Expectancy (Years)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "life_expectancy_distribution.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    # Task 9: Distribution of Mean Years of Schooling
    print("Task 9: Generating Mean Schooling Distribution Plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data1, x="Mean yrs of schooling", kde=True, color="#10b981", bins=10, fill=True, alpha=0.6)
    plt.title("Distribution of Mean Years of Schooling", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Mean Years of Schooling", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mean_schooling_distribution.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    # Task 10: GNI per capita by Development Tier (Strip Plot)
    print("Task 10: Generating GNI by Tier Strip Plot...")
    plt.figure(figsize=(8, 5))
    tier_order = ['Low', 'Medium', 'High', 'Very High']
    # Filter order to only what exists in head(20) to avoid plotting errors
    present_tiers = [t for t in tier_order if t in data1['Development Tier'].unique()]
    sns.stripplot(data=data1, x="Development Tier", y="GNI per capita", order=present_tiers,
                  palette=tier_colors, hue="Development Tier", jitter=0.2, size=8, alpha=0.8, legend=False)
    plt.yscale('log')
    plt.title("GNI Per Capita across Development Tiers", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Development Tier", fontsize=10)
    plt.ylabel("GNI Per Capita (PPP$, Log Scale)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strip_plot.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # Make a copy of mean schooling vs hdi scatter to match education_vs_hdi.png expected by index.html
    print("Saving education_vs_hdi.png as scatter task...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data1, x="Mean yrs of schooling", y="HDI Score", hue="Development Tier",
                    palette=tier_colors, s=100, alpha=0.9, edgecolor="#0b0f19")
    plt.title("Mean Years of Schooling vs HDI Score", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Mean Years of Schooling (Years)", fontsize=10)
    plt.ylabel("HDI Score", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "education_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()

    print("All 10 EDA visualizations successfully generated and saved to 'static/plots/'.")
    return True

if __name__ == "__main__":
    run_eda()
