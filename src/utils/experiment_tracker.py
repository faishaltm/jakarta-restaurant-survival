"""
Experiment Tracking for Hyperparameter Tuning
Logs all experiments, results, and configurations for easy debugging and comparison
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pickle
from loguru import logger


class ExperimentTracker:
    """Track experiments for hyperparameter tuning"""

    def __init__(self, experiment_name: str, output_dir: str = "outputs/experiments"):
        """
        Initialize experiment tracker

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.runs = []
        self.run_counter = 0
        self.best_run = None
        self.best_score = None

        logger.info(f"Experiment tracker initialized: {self.experiment_dir}")

    def log_run(
        self,
        run_id: int,
        strategy: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None,
        model_object: Any = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a single training run

        Args:
            run_id: Unique run identifier
            strategy: Imbalance handling strategy
            model_type: 'rsf' or 'gbsa'
            hyperparameters: Dictionary of hyperparameters used
            metrics: Dictionary of evaluation metrics
            feature_importance: DataFrame with feature importance
            model_object: Trained model object (will be pickled)
            metadata: Additional metadata
        """
        run_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'model_type': model_type,
            'hyperparameters': hyperparameters,
            'metrics': metrics,
            'metadata': metadata or {}
        }

        # Save to list
        self.runs.append(run_data)

        # Create run directory
        run_dir = self.experiment_dir / f"run_{run_id:04d}_{strategy}_{model_type}"
        run_dir.mkdir(exist_ok=True)

        # Save run configuration
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(run_data, f, indent=2)

        # Save feature importance
        if feature_importance is not None:
            feature_importance.to_csv(run_dir / 'feature_importance.csv', index=False)

        # Save model
        if model_object is not None:
            with open(run_dir / 'model.pkl', 'wb') as f:
                pickle.dump(model_object, f)

        # Update best run
        primary_metric = metrics.get('c_index_uno', metrics.get('c_index_harrell', 0))
        if self.best_score is None or primary_metric > self.best_score:
            self.best_score = primary_metric
            self.best_run = run_data
            logger.info(f"🏆 New best model! Run {run_id}: {primary_metric:.4f}")

        logger.debug(f"Logged run {run_id}: {strategy}/{model_type} - C-index (Uno): {metrics.get('c_index_uno', 'N/A'):.4f}")

    def save_summary(self):
        """Save experiment summary"""
        # Convert runs to DataFrame
        runs_flat = []
        for run in self.runs:
            flat_run = {
                'run_id': run['run_id'],
                'timestamp': run['timestamp'],
                'strategy': run['strategy'],
                'model_type': run['model_type']
            }

            # Add hyperparameters with prefix
            for key, value in run['hyperparameters'].items():
                flat_run[f'hp_{key}'] = value

            # Add metrics with prefix
            for key, value in run['metrics'].items():
                flat_run[f'metric_{key}'] = value

            runs_flat.append(flat_run)

        runs_df = pd.DataFrame(runs_flat)

        # Sort by primary metric
        if 'metric_c_index_uno' in runs_df.columns:
            runs_df = runs_df.sort_values('metric_c_index_uno', ascending=False)
        elif 'metric_c_index_harrell' in runs_df.columns:
            runs_df = runs_df.sort_values('metric_c_index_harrell', ascending=False)

        # Save
        runs_df.to_csv(self.experiment_dir / 'all_runs.csv', index=False)
        logger.info(f"Saved all runs: {self.experiment_dir / 'all_runs.csv'}")

        # Save best run
        if self.best_run:
            with open(self.experiment_dir / 'best_run.json', 'w') as f:
                json.dump(self.best_run, f, indent=2)
            logger.info(f"Best run saved: {self.experiment_dir / 'best_run.json'}")

        # Create summary report
        summary = {
            'experiment_name': self.experiment_name,
            'total_runs': len(self.runs),
            'best_run_id': self.best_run['run_id'] if self.best_run else None,
            'best_score': self.best_score,
            'best_strategy': self.best_run['strategy'] if self.best_run else None,
            'best_model_type': self.best_run['model_type'] if self.best_run else None,
            'experiment_dir': str(self.experiment_dir)
        }

        with open(self.experiment_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return runs_df, summary

    def get_best_hyperparameters(self, model_type: str = None, strategy: str = None) -> Dict:
        """
        Get best hyperparameters, optionally filtered by model type or strategy

        Args:
            model_type: Filter by model type ('rsf' or 'gbsa')
            strategy: Filter by imbalance strategy

        Returns:
            Dictionary of best hyperparameters
        """
        filtered_runs = self.runs

        if model_type:
            filtered_runs = [r for r in filtered_runs if r['model_type'] == model_type]

        if strategy:
            filtered_runs = [r for r in filtered_runs if r['strategy'] == strategy]

        if not filtered_runs:
            return {}

        # Find best by primary metric
        best = max(
            filtered_runs,
            key=lambda r: r['metrics'].get('c_index_uno', r['metrics'].get('c_index_harrell', 0))
        )

        return best['hyperparameters']

    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all strategies

        Returns:
            DataFrame with strategy comparison
        """
        strategy_metrics = []

        for run in self.runs:
            strategy_metrics.append({
                'strategy': run['strategy'],
                'model_type': run['model_type'],
                'c_index_uno': run['metrics'].get('c_index_uno'),
                'c_index_harrell': run['metrics'].get('c_index_harrell'),
                'ibs': run['metrics'].get('integrated_brier_score'),
                'auc_6mo': run['metrics'].get('auc_6mo'),
                'auc_1yr': run['metrics'].get('auc_1yr'),
                'auc_2yr': run['metrics'].get('auc_2yr')
            })

        df = pd.DataFrame(strategy_metrics)

        # Group by strategy and model_type, get mean
        grouped = df.groupby(['strategy', 'model_type']).mean().reset_index()

        grouped.to_csv(self.experiment_dir / 'strategy_comparison.csv', index=False)

        return grouped

    def plot_hyperparameter_impact(self, hyperparameter: str, metric: str = 'c_index_uno'):
        """
        Plot impact of a hyperparameter on performance

        Args:
            hyperparameter: Hyperparameter name
            metric: Metric to plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = []
        for run in self.runs:
            if hyperparameter in run['hyperparameters']:
                data.append({
                    'hyperparameter_value': run['hyperparameters'][hyperparameter],
                    'metric_value': run['metrics'].get(metric),
                    'model_type': run['model_type'],
                    'strategy': run['strategy']
                })

        if not data:
            logger.warning(f"No data found for hyperparameter: {hyperparameter}")
            return

        df = pd.DataFrame(data)

        plt.figure(figsize=(12, 6))
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            plt.scatter(
                model_data['hyperparameter_value'],
                model_data['metric_value'],
                label=model_type,
                alpha=0.6,
                s=100
            )

        plt.xlabel(hyperparameter, fontweight='bold')
        plt.ylabel(metric, fontweight='bold')
        plt.title(f'Impact of {hyperparameter} on {metric}', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.experiment_dir / f'hp_impact_{hyperparameter}_{metric}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved hyperparameter impact plot: {output_path}")


class ExperimentLogger:
    """Simple logger for debugging experiments"""

    def __init__(self, log_file: str):
        """
        Initialize experiment logger

        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Clear existing log
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment Log - {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        # Also print to console
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "DEBUG":
            logger.debug(message)
        else:
            logger.info(message)

    def log_error(self, message: str, exception: Exception = None):
        """
        Log an error with traceback

        Args:
            message: Error message
            exception: Exception object
        """
        self.log(f"ERROR: {message}", level="ERROR")
        if exception:
            import traceback
            tb = traceback.format_exc()
            with open(self.log_file, 'a') as f:
                f.write(f"{tb}\n")


if __name__ == "__main__":
    # Test experiment tracker
    tracker = ExperimentTracker("test_experiment")

    # Log some test runs
    for i in range(5):
        tracker.log_run(
            run_id=i,
            strategy='hybrid',
            model_type='rsf',
            hyperparameters={'n_estimators': 100 + i*50, 'max_depth': 10 + i},
            metrics={
                'c_index_uno': 0.75 + i*0.01,
                'c_index_harrell': 0.73 + i*0.01,
                'integrated_brier_score': 0.15 - i*0.01
            }
        )

    # Save summary
    runs_df, summary = tracker.save_summary()

    print("\nAll runs:")
    print(runs_df)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print(f"\nExperiment directory: {tracker.experiment_dir}")
