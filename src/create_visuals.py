import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration --- #
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results_testing"
VISUALS_DIR = ROOT_DIR / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

BACKTEST_RESULTS_PATH = RESULTS_DIR / "backtest_results.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
FORMAL_TEST_RESULTS_PATH = RESULTS_DIR / "formal_test_results.csv"

def create_summary_table(df_backtest, df_formal_tests):
    """Creates a summary table of the backtest results including formal tests."""
    baseline_breaches = df_backtest['baseline_breach'].sum()
    dynamic_breaches = df_backtest['dynamic_breach'].sum()
    avg_baseline_margin = (df_backtest['baseline_margin'] / df_backtest['price']).mean()
    avg_dynamic_margin = (df_backtest['dynamic_margin'] / df_backtest['price']).mean()
    pro_baseline = (df_backtest['baseline_margin'] / df_backtest['price']).diff().std()
    pro_dynamic = (df_backtest['dynamic_margin'] / df_backtest['price']).diff().std()

    summary_data = {
        'Metric': ['Breach Rate (%)', 'Number of Breaches', 'Average Margin Size (% of Price)', 'Procyclicality'],
        'Baseline Model': [
            f"{baseline_breaches / len(df_backtest):.2%}",
            baseline_breaches,
            f"{avg_baseline_margin:.2%}",
            f"{pro_baseline:.4f}"
        ],
        'Dynamic Model': [
            f"{dynamic_breaches / len(df_backtest):.2%}",
            dynamic_breaches,
            f"{avg_dynamic_margin:.2%}",
            f"{pro_dynamic:.4f}"
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    # Add formal test results
    uc_p_baseline = df_formal_tests.loc[('baseline_990', 'UC_p_value'), '0']
    ind_p_baseline = df_formal_tests.loc[('baseline_990', 'Ind_p_value'), '0']
    uc_p_dynamic = df_formal_tests.loc[('dynamic_990', 'UC_p_value'), '0']
    ind_p_dynamic = df_formal_tests.loc[('dynamic_990', 'Ind_p_value'), '0']

    df_formal_summary = pd.DataFrame({
        'Metric': ['Kupiec UC p-value', 'Christoffersen Ind p-value'],
        'Baseline Model': [f"{uc_p_baseline:.4f}", f"{ind_p_baseline:.4f}"],
        'Dynamic Model': [f"{uc_p_dynamic:.4f}", f"{ind_p_dynamic:.4f}"]
    })
    df_summary = pd.concat([df_summary, df_formal_summary], ignore_index=True)

    print("--- Performance Summary Table ---")
    print(df_summary.to_markdown(index=False))
    df_summary.to_csv(VISUALS_DIR / "performance_summary.csv", index=False)

def create_breach_analysis_table(df_backtest):
    """Creates a table of the breach analysis."""
    baseline_breach_dates = df_backtest[df_backtest['baseline_breach'] == 1].index
    dynamic_breach_dates = df_backtest[df_backtest['dynamic_breach'] == 1].index

    df_baseline_breaches = df_backtest.loc[baseline_breach_dates, ['price', 'pnl', 'baseline_margin', 'garch_volatility']]
    df_dynamic_breaches = df_backtest.loc[dynamic_breach_dates, ['price', 'pnl', 'dynamic_margin', 'im_correction_factor']]

    print("\n--- Baseline Breach Analysis ---")
    print(df_baseline_breaches.to_markdown())
    df_baseline_breaches.to_csv(VISUALS_DIR / "baseline_breach_analysis.csv")

    print("\n--- Dynamic Breach Analysis ---")
    print(df_dynamic_breaches.to_markdown())
    df_dynamic_breaches.to_csv(VISUALS_DIR / "dynamic_breach_analysis.csv")

def create_margin_comparison_graph(df_backtest):
    """
    Creates a graph comparing the baseline and dynamic margins, with breach points.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_backtest.index, df_backtest['baseline_margin'], label='Baseline Margin', color='blue', alpha=0.7)
    ax.plot(df_backtest.index, df_backtest['dynamic_margin'], label='Dynamic Margin', color='red', alpha=0.7)
    ax.plot(df_backtest.index, df_backtest['pnl'].abs(), label='Absolute P&L', color='green', alpha=0.5, linestyle='--')

    # Add breach markers
    baseline_breaches = df_backtest[df_backtest['baseline_breach'] == 1]
    dynamic_breaches = df_backtest[df_backtest['dynamic_breach'] == 1]

    ax.scatter(baseline_breaches.index, baseline_breaches['pnl'].abs(),
               marker='o', color='blue', s=100, zorder=5, label='Baseline Breach')
    ax.scatter(dynamic_breaches.index, dynamic_breaches['pnl'].abs(),
               marker='x', color='red', s=100, zorder=5, label='Dynamic Breach')

    ax.set_title('Margin Comparison: Baseline vs. Dynamic with Breaches')
    ax.set_xlabel('Date')
    ax.set_ylabel('Margin / P&L')
    ax.legend()
    ax.grid(True)

    plt.savefig(VISUALS_DIR / "margin_comparison_with_breaches.png")
    print("\nMargin comparison graph with breaches saved to visuals/margin_comparison_with_breaches.png")

def create_feature_importance_graph(df_feature_importance):
    """Creates a graph of the feature importances."""
    df_top_15 = df_feature_importance.head(15)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(x='importance', y='feature', data=df_top_15, ax=ax, palette='viridis')

    ax.set_title('Top 15 Most Important Features for XGBoost Model')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png")
    print("\nFeature importance graph saved to visuals/feature_importance.png")

def create_im_correction_factor_time_series_graph(df_backtest):
    """Creates a time-series graph of the IM correction factor."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_backtest.index, df_backtest['im_correction_factor'], label='IM Correction Factor', color='purple', alpha=0.7)

    ax.set_title('IM Correction Factor Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('IM Correction Factor')
    ax.legend()
    ax.grid(True)

    plt.savefig(VISUALS_DIR / "im_correction_factor_time_series.png")
    print("\nIM correction factor time series graph saved to visuals/im_correction_factor_time_series.png")

def create_reliability_curve(df_backtest):
    """Creates a reliability curve."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    confidence_levels = [0.95, 0.99, 0.995]
    observed_breach_rates = []
    expected_breach_rates = []

    for cl in confidence_levels:
        n_breaches = df_backtest[f'baseline_breach_{int(cl*1000)}'].sum()
        breach_rate = n_breaches / len(df_backtest)
        observed_breach_rates.append(breach_rate)
        expected_breach_rates.append(1 - cl)

    ax.plot([0, 0.05], [0, 0.05], linestyle='--', color='gray', label='Perfect Reliability')
    ax.plot(expected_breach_rates, observed_breach_rates, marker='o', linestyle='-', color='blue', label='Observed Reliability')

    ax.set_title('Reliability Curve (Baseline Model)')
    ax.set_xlabel('Expected Breach Rate (1 - Confidence Level)')
    ax.set_ylabel('Observed Breach Rate')
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 0.05)
    ax.legend()
    ax.grid(True)

    plt.savefig(VISUALS_DIR / "reliability_curve.png")
    print("\nReliability curve saved to visuals/reliability_curve.png")

def create_phi_summary_table(df_backtest, df_formal_tests):
    """
    Computes and prints a summary table for the IM correction factor (phi_t).
    Includes mean, percentiles, and fraction of days at clipping limits.
    """
    phi_t = df_backtest['im_correction_factor']
    
    mean_phi = phi_t.mean()
    percentiles = phi_t.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    
    # Clipping limits
    F_min = 0.7
    F_max = 1.5
    
    # Fraction of days at clipping limits
    fraction_at_F_min = (phi_t <= F_min).sum() / len(phi_t)
    fraction_at_F_max = (phi_t >= F_max).sum() / len(phi_t)

    summary_data = {
        'Statistic': ['Mean', '1st Percentile', '5th Percentile', '25th Percentile', 
                      '50th Percentile (Median)', '75th Percentile', '95th Percentile', 
                      '99th Percentile', f'Fraction at F_min ({F_min})', f'Fraction at F_max ({F_max})'],
        'Value': [
            f"{mean_phi:.4f}",
            f"{percentiles[0.01]:.4f}",
            f"{percentiles[0.05]:.4f}",
            f"{percentiles[0.25]:.4f}",
            f"{percentiles[0.50]:.4f}",
            f"{percentiles[0.75]:.4f}",
            f"{percentiles[0.95]:.4f}",
            f"{percentiles[0.99]:.4f}",
            f"{fraction_at_F_min:.2%}",
            f"{fraction_at_F_max:.2%}"
        ]
    }
    df_phi_summary = pd.DataFrame(summary_data)
    
    print("\n--- IM Correction Factor (phi_t) Summary ---")
    print(df_phi_summary.to_markdown(index=False))
    df_phi_summary.to_csv(VISUALS_DIR / "phi_t_summary.csv", index=False)

def main():
    """Main function to generate all visuals."""
    # Load data
    df_backtest = pd.read_csv(BACKTEST_RESULTS_PATH, parse_dates=['date'], index_col='date')
    df_feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    df_formal_tests = pd.read_csv(FORMAL_TEST_RESULTS_PATH, index_col=[0, 1])

    # Generate visuals
    create_summary_table(df_backtest, df_formal_tests)
    create_breach_analysis_table(df_backtest)
    create_margin_comparison_graph(df_backtest)
    create_feature_importance_graph(df_feature_importance)
    create_im_correction_factor_time_series_graph(df_backtest)
    create_reliability_curve(df_backtest)
    create_phi_summary_table(df_backtest, df_formal_tests)


if __name__ == '__main__':
    main()
