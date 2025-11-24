"""
Survival Analysis Visualization Module

Static matplotlib visualizations for survival analysis results.

Author: Claude Code
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SurvivalVisualizer:
    """
    Create static matplotlib visualizations for survival analysis.
    """

    def __init__(self, output_dir: str = "outputs/survival_plots"):
        """
        Initialize survival visualizer.

        Args:
            output_dir: Directory to save plots (default: "outputs/survival_plots")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SurvivalVisualizer initialized. Plots will be saved to: {self.output_dir}")

    def plot_survival_curves(
        self,
        survival_functions: List,
        time_points: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Survival Probability Curves",
        percentiles: List[float] = [0.25, 0.5, 0.75],
        save_name: str = "survival_curves.png"
    ):
        """
        Plot survival probability curves over time.

        Args:
            survival_functions: List of survival functions from model.predict_survival_function()
            time_points: Array of time points to evaluate
            labels: Optional labels for different groups
            title: Plot title
            percentiles: Percentiles to highlight (default: [0.25, 0.5, 0.75])
            save_name: Filename to save plot
        """
        logger.info(f"Creating survival curves plot: {save_name}")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Calculate survival probabilities for each time point
        surv_probs = np.array([[fn(t) for t in time_points] for fn in survival_functions])

        # Plot mean survival curve
        mean_surv = surv_probs.mean(axis=0)
        ax.plot(time_points / 365, mean_surv, linewidth=2.5, label='Mean', color='darkblue')

        # Plot confidence intervals
        for pct in percentiles:
            pct_surv = np.percentile(surv_probs, pct * 100, axis=0)
            ax.plot(time_points / 365, pct_surv, linestyle='--', alpha=0.7,
                   label=f'{int(pct*100)}th percentile')

        # Fill between percentiles
        lower = np.percentile(surv_probs, 25, axis=0)
        upper = np.percentile(surv_probs, 75, axis=0)
        ax.fill_between(time_points / 365, lower, upper, alpha=0.2, color='blue',
                        label='25th-75th percentile')

        ax.set_xlabel('Time (Years)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Add horizontal reference lines
        for p in [0.25, 0.5, 0.75]:
            ax.axhline(y=p, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_name: str = "model_comparison.png"
    ):
        """
        Plot comparison of different models' performance.

        Args:
            comparison_df: DataFrame with model comparison metrics
            save_name: Filename to save plot
        """
        logger.info(f"Creating model comparison plot: {save_name}")

        # Prepare metrics for plotting
        metrics_to_plot = ['c_index']

        # Add AUC columns if they exist
        auc_cols = [col for col in comparison_df.columns if col.startswith('auc_')]
        if auc_cols:
            metrics_to_plot.extend(auc_cols)

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Extract metric values
            models = comparison_df['model'].values
            values = comparison_df[metric].values

            # Create bar plot
            bars = ax.bar(range(len(models)), values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])

            # Customize
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            # Set y-axis limits
            if metric == 'c_index' or metric.startswith('auc'):
                ax.set_ylim([0.5, 1.0])
                ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Good (0.7)')
                ax.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (0.8)')
                ax.legend(fontsize=8)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_name: str = "feature_importance.png"
    ):
        """
        Plot feature importance from survival model.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            title: Plot title
            save_name: Filename to save plot
        """
        logger.info(f"Creating feature importance plot: {save_name}")

        # Take top N features
        plot_df = importance_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

        # Create horizontal bar plot
        y_pos = np.arange(len(plot_df))
        bars = ax.barh(y_pos, plot_df['importance'].values, color='steelblue')

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['feature'].values, fontsize=9)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, plot_df['importance'].values)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()

    def plot_risk_stratification(
        self,
        risk_scores: np.ndarray,
        y: np.ndarray,
        n_groups: int = 4,
        title: str = "Risk Stratification",
        save_name: str = "risk_stratification.png"
    ):
        """
        Plot survival by risk groups.

        Args:
            risk_scores: Risk scores from model.predict()
            y: Structured array with event and duration
            n_groups: Number of risk groups (default: 4 = quartiles)
            title: Plot title
            save_name: Filename to save plot
        """
        logger.info(f"Creating risk stratification plot: {save_name}")

        # Create risk groups based on quartiles
        risk_groups = pd.qcut(risk_scores, q=n_groups, labels=False, duplicates='drop')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Event rate by risk group
        group_data = []
        for group in range(n_groups):
            mask = risk_groups == group
            if mask.sum() > 0:
                event_rate = y[mask]['event'].sum() / mask.sum()
                group_data.append({
                    'group': f'Q{group+1}',
                    'event_rate': event_rate,
                    'count': mask.sum()
                })

        group_df = pd.DataFrame(group_data)

        bars = ax1.bar(group_df['group'], group_df['event_rate'], color='coral')
        ax1.set_xlabel('Risk Group', fontsize=12)
        ax1.set_ylabel('Event Rate (Closure)', fontsize=12)
        ax1.set_title('Event Rate by Risk Group', fontsize=13, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bar, val, count in zip(bars, group_df['event_rate'], group_df['count']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}\n(n={count})', ha='center', va='bottom', fontsize=9)

        # Plot 2: Distribution of risk scores
        ax2.hist(risk_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Risk Score', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Risk Scores', fontsize=13, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)

        # Add vertical lines for quartiles
        for q in range(1, n_groups):
            quantile_val = np.percentile(risk_scores, q * 100 / n_groups)
            ax2.axvline(quantile_val, color='red', linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()

    def plot_time_point_predictions(
        self,
        predictions_df: pd.DataFrame,
        time_points: List[int],
        labels: Optional[pd.Series] = None,
        save_name: str = "time_point_predictions.png"
    ):
        """
        Plot survival probability predictions at specific time points.

        Args:
            predictions_df: DataFrame with survival probabilities at different time points
            time_points: List of time points (in days)
            labels: Optional true labels for comparison
            save_name: Filename to save plot
        """
        logger.info(f"Creating time point predictions plot: {save_name}")

        n_timepoints = len(time_points)
        fig, axes = plt.subplots(1, n_timepoints, figsize=(5 * n_timepoints, 5))

        if n_timepoints == 1:
            axes = [axes]

        for idx, t in enumerate(time_points):
            ax = axes[idx]
            col = f'surv_prob_{t}d'

            if col in predictions_df.columns:
                probs = predictions_df[col].values

                # Create histogram
                ax.hist(probs, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Survival Probability', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title(f'{t} Days ({t/365:.1f} Years)', fontsize=12, fontweight='bold')
                ax.grid(True, axis='y', alpha=0.3)

                # Add mean line
                mean_prob = probs.mean()
                ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.3f}')
                ax.legend()

                # Add statistics text
                stats_text = f'Mean: {probs.mean():.3f}\nMedian: {np.median(probs):.3f}\nStd: {probs.std():.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()

    def plot_category_survival_comparison(
        self,
        df: pd.DataFrame,
        category_col: str = 'main_category',
        top_n: int = 10,
        save_name: str = "category_survival_comparison.png"
    ):
        """
        Compare survival rates across different POI categories.

        Args:
            df: DataFrame with category labels and survival data
            category_col: Column name for category
            top_n: Number of top categories to show
            save_name: Filename to save plot
        """
        logger.info(f"Creating category survival comparison plot: {save_name}")

        # Calculate statistics by category
        category_stats = []

        top_categories = df[category_col].value_counts().head(top_n).index

        for cat in top_categories:
            cat_df = df[df[category_col] == cat]

            if len(cat_df) >= 10:  # Only include categories with sufficient data
                stats = {
                    'category': cat,
                    'count': len(cat_df),
                    'event_rate': cat_df['event'].sum() / len(cat_df),
                    'mean_duration': cat_df['duration'].mean() / 365,  # Convert to years
                    'success_rate': (cat_df['categorical_label'] == 1).sum() / len(cat_df)
                }
                category_stats.append(stats)

        stats_df = pd.DataFrame(category_stats).sort_values('event_rate', ascending=False)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Event Rate (Closure Rate)
        ax1.barh(range(len(stats_df)), stats_df['event_rate'].values, color='salmon')
        ax1.set_yticks(range(len(stats_df)))
        ax1.set_yticklabels(stats_df['category'].values, fontsize=9)
        ax1.set_xlabel('Event Rate (Closure)', fontsize=11)
        ax1.set_title('Closure Rate by Category', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(stats_df['event_rate'].values):
            ax1.text(val + 0.005, i, f'{val:.1%}', va='center', fontsize=8)

        # Plot 2: Mean Duration
        ax2.barh(range(len(stats_df)), stats_df['mean_duration'].values, color='skyblue')
        ax2.set_yticks(range(len(stats_df)))
        ax2.set_yticklabels(stats_df['category'].values, fontsize=9)
        ax2.set_xlabel('Mean Duration (Years)', fontsize=11)
        ax2.set_title('Average Business Duration', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(stats_df['mean_duration'].values):
            ax2.text(val + 0.2, i, f'{val:.1f}y', va='center', fontsize=8)

        # Plot 3: Success Rate (>=2 years)
        ax3.barh(range(len(stats_df)), stats_df['success_rate'].values, color='lightgreen')
        ax3.set_yticks(range(len(stats_df)))
        ax3.set_yticklabels(stats_df['category'].values, fontsize=9)
        ax3.set_xlabel('Success Rate (>=2 years)', fontsize=11)
        ax3.set_title('2-Year Success Rate', fontsize=12, fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(stats_df['success_rate'].values):
            ax3.text(val + 0.01, i, f'{val:.1%}', va='center', fontsize=8)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()

    def create_summary_report(
        self,
        model_results: Dict,
        importance_df: pd.DataFrame,
        save_name: str = "summary_report.png"
    ):
        """
        Create a comprehensive summary report with multiple panels.

        Args:
            model_results: Dictionary with model evaluation results
            importance_df: DataFrame with feature importance
            save_name: Filename to save plot
        """
        logger.info(f"Creating summary report: {save_name}")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Model Performance Metrics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        metrics_text = f"""
        MODEL PERFORMANCE SUMMARY
        {'=' * 80}
        Model: {model_results.get('model_name', 'N/A')}

        Primary Metrics:
        - C-index (Concordance Index): {model_results.get('c_index', 0):.4f}
          {'  -> Excellent!' if model_results.get('c_index', 0) > 0.8 else '  -> Good' if model_results.get('c_index', 0) > 0.7 else '  -> Fair'}

        - Integrated Brier Score: {model_results.get('integrated_brier_score', 'N/A')}
          (Lower is better, 0.25 = random baseline)

        Time-dependent AUC:
        """

        if 'time_dependent_auc' in model_results:
            for time_label, auc in model_results['time_dependent_auc'].items():
                if auc is not None:
                    metrics_text += f"\n        - {time_label}: {auc:.4f}"

        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Panel 2: Top Features
        ax2 = fig.add_subplot(gs[1:, :])
        top_features = importance_df.head(15)
        y_pos = np.arange(len(top_features))

        ax2.barh(y_pos, top_features['importance'].values, color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Feature Importance', fontsize=11)
        ax2.set_title('Top 15 Most Important Features', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        plt.close()
