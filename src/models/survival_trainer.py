"""
Survival Analysis Model Trainer

This module handles training and evaluation of survival analysis models:
- Random Survival Forest (RSF)
- Gradient Boosting Survival Analysis (GBSA)

Author: Claude Code
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc
)
from sksurv.util import Surv
import joblib

logger = logging.getLogger(__name__)


class SurvivalModelTrainer:
    """
    Trainer for survival analysis models.

    Supports:
    - Random Survival Forest
    - Gradient Boosting Survival Analysis
    - Multiple evaluation metrics
    - Time-dependent predictions
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize survival model trainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.feature_names = None
        self.training_results = {}

        logger.info(f"SurvivalModelTrainer initialized (random_state={random_state})")

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_event_col: str = 'event',
        target_duration_col: str = 'duration',
        test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for survival model training.

        Args:
            df: DataFrame with features and survival labels
            feature_cols: List of feature column names (if None, auto-detect numeric columns)
            target_event_col: Column name for event indicator
            target_duration_col: Column name for duration
            test_size: Proportion of data for testing (default: 0.3)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
            where y_train and y_test are structured arrays
        """
        logger.info("Preparing data for survival analysis...")

        # Auto-detect feature columns if not provided
        if feature_cols is None:
            # Exclude non-numeric columns and target columns
            exclude_cols = [
                target_event_col, target_duration_col,
                'geometry', 'fsq_place_id', 'name', 'address',
                'locality', 'region', 'postcode', 'country',
                'fsq_category_ids', 'fsq_category_labels',
                'primary_category', 'categories_list', 'category_count',
                'date_created', 'date_refreshed', 'date_closed',
                'date_created_parsed', 'date_closed_parsed',
                'categorical_label', 'poi_type', 'is_coffee_shop',
                'category_str'
            ]

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            feature_cols = [
                col for col in numeric_cols
                if col not in exclude_cols and not col.endswith('_parsed')
            ]

            logger.info(f"Auto-detected {len(feature_cols)} numeric feature columns")

        self.feature_names = feature_cols

        # Extract features
        X = df[feature_cols].values

        # Check for missing values
        missing_mask = np.isnan(X).any(axis=1)
        if missing_mask.any():
            n_missing = missing_mask.sum()
            logger.warning(f"Found {n_missing} rows with missing values, will be removed")
            valid_mask = ~missing_mask
            X = X[valid_mask]
            df = df[valid_mask].copy()

        # Create structured array for survival target
        y = Surv.from_dataframe(target_event_col, target_duration_col, df)

        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Event rate: {df[target_event_col].sum() / len(df) * 100:.2f}%")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df[target_event_col]  # Stratify by event status
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_random_survival_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        **kwargs
    ) -> RandomSurvivalForest:
        """
        Train Random Survival Forest model.

        Args:
            X_train: Training features
            y_train: Training target (structured array)
            n_estimators: Number of trees (default: 100)
            max_depth: Maximum tree depth (default: None = unlimited)
            min_samples_split: Minimum samples to split node (default: 10)
            min_samples_leaf: Minimum samples in leaf (default: 5)
            n_jobs: Number of parallel jobs (default: -1 = all CPUs)
            **kwargs: Additional parameters for RandomSurvivalForest

        Returns:
            Trained RandomSurvivalForest model
        """
        logger.info("Training Random Survival Forest...")
        logger.info(f"  - n_estimators: {n_estimators}")
        logger.info(f"  - max_depth: {max_depth}")
        logger.info(f"  - min_samples_split: {min_samples_split}")
        logger.info(f"  - min_samples_leaf: {min_samples_leaf}")

        model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=self.random_state,
            **kwargs
        )

        model.fit(X_train, y_train)

        self.models['random_survival_forest'] = model
        logger.info("Random Survival Forest training completed!")

        return model

    def train_gradient_boosting_survival(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        **kwargs
    ) -> GradientBoostingSurvivalAnalysis:
        """
        Train Gradient Boosting Survival Analysis model.

        Args:
            X_train: Training features
            y_train: Training target (structured array)
            n_estimators: Number of boosting stages (default: 100)
            learning_rate: Learning rate (default: 0.1)
            max_depth: Maximum tree depth (default: 3)
            min_samples_split: Minimum samples to split node (default: 10)
            min_samples_leaf: Minimum samples in leaf (default: 5)
            subsample: Fraction of samples for fitting base learners (default: 1.0)
            **kwargs: Additional parameters for GradientBoostingSurvivalAnalysis

        Returns:
            Trained GradientBoostingSurvivalAnalysis model
        """
        logger.info("Training Gradient Boosting Survival Analysis...")
        logger.info(f"  - n_estimators: {n_estimators}")
        logger.info(f"  - learning_rate: {learning_rate}")
        logger.info(f"  - max_depth: {max_depth}")
        logger.info(f"  - subsample: {subsample}")

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=self.random_state,
            **kwargs
        )

        model.fit(X_train, y_train)

        self.models['gradient_boosting_survival'] = model
        logger.info("Gradient Boosting Survival Analysis training completed!")

        return model

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_train: np.ndarray,
        time_points: List[int] = [180, 365, 730, 1095]
    ) -> Dict:
        """
        Evaluate survival model with multiple metrics.

        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target (structured array)
            y_train: Training target (for IBS calculation)
            time_points: Time points for evaluation (in days)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        model = self.models[model_name]
        results = {'model_name': model_name}

        # 1. Concordance Index (C-index)
        # Measures ranking ability: 0.5=random, >0.7=good, >0.8=excellent
        risk_scores = model.predict(X_test)
        c_index = concordance_index_censored(
            y_test['event'],
            y_test['duration'],
            risk_scores
        )
        results['c_index'] = float(c_index[0])

        logger.info(f"C-index: {results['c_index']:.4f}")
        if results['c_index'] > 0.8:
            logger.info("  -> Excellent discrimination!")
        elif results['c_index'] > 0.7:
            logger.info("  -> Good discrimination")
        elif results['c_index'] > 0.6:
            logger.info("  -> Fair discrimination")
        else:
            logger.info("  -> Poor discrimination")

        # 2. Integrated Brier Score (IBS)
        # Measures calibration: lower is better (0=perfect, 0.25=random)
        try:
            # Get survival functions
            surv_funcs = model.predict_survival_function(X_test)

            # Calculate IBS over time points
            times = np.array(time_points)

            # Get survival probabilities at specific times
            preds = np.array([[fn(t) for t in times] for fn in surv_funcs])

            # Calculate Brier scores
            from sksurv.metrics import brier_score
            brier_scores = []
            for i, t in enumerate(times):
                # Only calculate if we have events at or before this time
                if (y_train['duration'] >= t).any():
                    bs = brier_score(y_train, y_test, preds[:, i], times=t)
                    brier_scores.append(bs[1])
                else:
                    brier_scores.append(np.nan)

            results['brier_scores'] = {
                f'{t}_days': float(bs) if not np.isnan(bs) else None
                for t, bs in zip(time_points, brier_scores)
            }

            # Calculate mean IBS (excluding NaN)
            valid_scores = [bs for bs in brier_scores if not np.isnan(bs)]
            if valid_scores:
                results['integrated_brier_score'] = float(np.mean(valid_scores))
                logger.info(f"Integrated Brier Score: {results['integrated_brier_score']:.4f}")
            else:
                results['integrated_brier_score'] = None
                logger.warning("Could not calculate IBS (insufficient events)")

        except Exception as e:
            logger.warning(f"Could not calculate Brier Score: {e}")
            results['brier_scores'] = {}
            results['integrated_brier_score'] = None

        # 3. Time-dependent AUC
        # AUC at specific time points
        try:
            auc_scores = {}
            for t in time_points:
                # Only calculate if we have enough events
                events_before_t = (y_test['duration'] <= t) & y_test['event']
                if events_before_t.sum() >= 5:  # Need at least 5 events
                    auc, mean_auc = cumulative_dynamic_auc(
                        y_train, y_test, risk_scores, times=[t]
                    )
                    auc_scores[f'{t}_days'] = float(auc[0])
                else:
                    auc_scores[f'{t}_days'] = None

            results['time_dependent_auc'] = auc_scores

            # Log AUC scores
            logger.info("Time-dependent AUC:")
            for time_label, auc_val in auc_scores.items():
                if auc_val is not None:
                    logger.info(f"  - {time_label}: {auc_val:.4f}")
                else:
                    logger.info(f"  - {time_label}: N/A (insufficient events)")

        except Exception as e:
            logger.warning(f"Could not calculate time-dependent AUC: {e}")
            results['time_dependent_auc'] = {}

        # 4. Survival probability predictions at time points
        results['time_points_evaluated'] = time_points

        # Store results
        self.training_results[model_name] = results

        return results

    def predict_survival_probability(
        self,
        model_name: str,
        X: np.ndarray,
        time_points: List[int] = [180, 365, 730, 1095]
    ) -> pd.DataFrame:
        """
        Predict survival probabilities at specific time points.

        Args:
            model_name: Name of the trained model
            X: Features to predict on
            time_points: Time points in days (default: [180, 365, 730, 1095])

        Returns:
            DataFrame with survival probabilities for each time point
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        model = self.models[model_name]

        logger.info(f"Predicting survival probabilities with {model_name}...")
        logger.info(f"Time points: {time_points} days")

        # Get survival functions
        surv_funcs = model.predict_survival_function(X)

        # Extract probabilities at specific time points
        predictions = {}
        for t in time_points:
            prob_col = f'surv_prob_{t}d'
            predictions[prob_col] = [fn(t) for fn in surv_funcs]

        pred_df = pd.DataFrame(predictions)

        logger.info(f"Predictions shape: {pred_df.shape}")
        logger.info(f"\nSurvival probability statistics:")
        for col in pred_df.columns:
            logger.info(f"  {col}: mean={pred_df[col].mean():.3f}, median={pred_df[col].median():.3f}")

        return pred_df

    def get_feature_importance(
        self,
        model_name: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model_name: Name of the trained model
            top_n: Number of top features to return (default: 20)

        Returns:
            DataFrame with feature names and importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        model = self.models[model_name]

        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        logger.info(f"\nTop {min(top_n, len(importance_df))} features for {model_name}:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def save_model(
        self,
        model_name: str,
        output_dir: str = "models/survival"
    ):
        """
        Save trained model to disk.

        Args:
            model_name: Name of the model to save
            output_dir: Directory to save model (default: "models/survival")
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f"{model_name}.joblib"
        joblib.dump(self.models[model_name], model_file)
        logger.info(f"Model saved: {model_file}")

        # Save feature names
        features_file = output_path / f"{model_name}_features.json"
        with open(features_file, 'w') as f:
            json.dump({'feature_names': self.feature_names}, f, indent=2)
        logger.info(f"Features saved: {features_file}")

        # Save training results if available
        if model_name in self.training_results:
            results_file = output_path / f"{model_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.training_results[model_name], f, indent=2)
            logger.info(f"Results saved: {results_file}")

    def load_model(
        self,
        model_name: str,
        model_dir: str = "models/survival"
    ):
        """
        Load trained model from disk.

        Args:
            model_name: Name of the model to load
            model_dir: Directory containing saved model (default: "models/survival")
        """
        model_path = Path(model_dir)

        # Load model
        model_file = model_path / f"{model_name}.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.models[model_name] = joblib.load(model_file)
        logger.info(f"Model loaded: {model_file}")

        # Load feature names
        features_file = model_path / f"{model_name}_features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                data = json.load(f)
                self.feature_names = data['feature_names']
            logger.info(f"Features loaded: {len(self.feature_names)} features")

        # Load training results if available
        results_file = model_path / f"{model_name}_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.training_results[model_name] = json.load(f)
            logger.info(f"Results loaded: {results_file}")

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models based on evaluation metrics.

        Returns:
            DataFrame comparing models
        """
        if not self.training_results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.training_results.items():
            row = {
                'model': model_name,
                'c_index': results.get('c_index'),
                'integrated_brier_score': results.get('integrated_brier_score'),
            }

            # Add time-dependent AUC scores
            if 'time_dependent_auc' in results:
                for time_label, auc in results['time_dependent_auc'].items():
                    row[f'auc_{time_label}'] = auc

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        logger.info(comparison_df.to_string(index=False))
        logger.info("=" * 80)

        return comparison_df
