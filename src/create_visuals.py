import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TABLES_DIR, FIGURES_DIR

def create_ablation_plot():
    """Creates and saves a bar plot of the ablation analysis results."""
    ablation_df = pd.read_csv(TABLES_DIR / "ablation_results.csv")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='performance_drop', y='removed_feature', data=ablation_df.head(15), palette='viridis')
    plt.title('Top 15 Feature Ablation Analysis')
    plt.xlabel('Performance Drop (R-squared)')
    plt.ylabel('Removed Feature')
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "ablation_analysis.png"
    plt.savefig(output_path)
    print(f"Ablation analysis plot saved to {output_path}")

def create_sensitivity_plot():
    """Creates and saves a line plot of the sensitivity analysis results."""
    sensitivity_df = pd.read_csv(TABLES_DIR / "sensitivity_analysis_iv_vol_ratio.csv")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='iv_vol_ratio', y='predicted_gamma', data=sensitivity_df)
    plt.title('Sensitivity Analysis: Predicted Gamma vs. IV Volume Ratio')
    plt.xlabel('IV Volume Ratio')
    plt.ylabel('Predicted Gamma')
    plt.grid(True)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "sensitivity_analysis_iv_vol_ratio.png"
    plt.savefig(output_path)
    print(f"Sensitivity analysis plot saved to {output_path}")

if __name__ == "__main__":
    create_ablation_plot()
    create_sensitivity_plot()
