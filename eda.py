import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Starting Exploratory Data Analysis...")
    local_processed_path = "hdi_processed.csv"
    plots_dir = os.path.join("static", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(local_processed_path):
        print(f"Error: Processed dataset '{local_processed_path}' not found! Run download_data.py first.")
        return False
        
    df = pd.read_csv(local_processed_path)
    
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
    
    # Custom palette matching development tiers
    tier_colors = {
        "Very High": "#06b6d4",  # Cyan
        "High": "#10b981",       # Emerald Green
        "Medium": "#f59e0b",     # Amber Orange
        "Low": "#ef4444"         # Rose Red
    }
    
    # ------------------ Plot 1: Correlation Heatmap ------------------
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    corr_cols = ['Life expectancy', 'Expected yrs of schooling', 'Mean yrs of schooling', 'GNI per capita', 'HDI Score']
    corr_matrix = df[corr_cols].corr()
    
    # Heatmap visualization
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", 
                cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5, linecolor="#0b0f19")
    plt.title("Correlation Matrix of HDI Component Indicators", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # ------------------ Plot 2: Distribution of HDI Score ------------------
    print("Generating HDI Score Distribution Plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="HDI Score", kde=True, color="#8b5cf6", bins=20, fill=True, alpha=0.6)
    plt.title("Global Distribution of Human Development Index (HDI) Scores", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("HDI Score", fontsize=12)
    plt.ylabel("Number of Countries", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hdi_distribution.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # ------------------ Plot 3: Scatter Plot - GNI vs HDI (Log Scale) ------------------
    print("Generating GNI vs HDI Scatter Plot...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="GNI per capita", y="HDI Score", hue="Development Tier", 
                    palette=tier_colors, alpha=0.8, edgecolor="#0b0f19", s=70)
    plt.xscale('log')
    plt.title("HDI Score vs Gross National Income (GNI) Per Capita (Log Scale)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("GNI Per Capita (PPP$, Log Scale)", fontsize=12)
    plt.ylabel("HDI Score", fontsize=12)
    plt.legend(title="Development Tier", title_fontsize='11', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gni_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # ------------------ Plot 4: Scatter Plot - Life Expectancy vs HDI ------------------
    print("Generating Life Expectancy vs HDI Scatter Plot...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="Life expectancy", y="HDI Score", hue="Development Tier", 
                    palette=tier_colors, alpha=0.8, edgecolor="#0b0f19", s=70)
    plt.title("HDI Score vs Life Expectancy at Birth", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Life Expectancy (Years)", fontsize=12)
    plt.ylabel("HDI Score", fontsize=12)
    plt.legend(title="Development Tier", title_fontsize='11', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "life_expectancy_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # ------------------ Plot 5: Scatter Plot - Mean Schooling vs HDI ------------------
    print("Generating Schooling vs HDI Scatter Plot...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="Mean yrs of schooling", y="HDI Score", hue="Development Tier", 
                    palette=tier_colors, alpha=0.8, edgecolor="#0b0f19", s=70)
    plt.title("HDI Score vs Mean Years of Schooling", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mean Years of Schooling (Years)", fontsize=12)
    plt.ylabel("HDI Score", fontsize=12)
    plt.legend(title="Development Tier", title_fontsize='11', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "education_vs_hdi.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    # ------------------ Plot 6: Strip Plot - GNI by Tier ------------------
    print("Generating Strip Plot...")
    plt.figure(figsize=(9, 5))
    # Order categories from Low to Very High
    tier_order = ['Low', 'Medium', 'High', 'Very High']
    sns.stripplot(data=df, x="Development Tier", y="GNI per capita", order=tier_order,
                  palette=tier_colors, hue="Development Tier", jitter=0.25, size=8, alpha=0.7, dodge=False, legend=False)
    plt.yscale('log')
    plt.title("GNI Per Capita Distribution across Development Tiers", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Development Tier (HDI Category)", fontsize=12)
    plt.ylabel("GNI Per Capita (PPP$, Log Scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strip_plot.png"), dpi=200, facecolor="#0b0f19")
    plt.close()
    
    print("EDA Visualizations successfully generated and saved to 'static/plots/'.")
    return True

if __name__ == "__main__":
    run_eda()
