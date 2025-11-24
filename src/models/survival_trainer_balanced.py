"""
Enhanced Survival Analysis Trainer with Imbalance Handling

Implements multiple strategies for handling highly imbalanced survival data:
1. Sample weighting (event-based weights)
2. IPCW (Inverse Probability of Censoring Weighting) loss
3. Random undersampling for Balanced RSF
4. Hybrid approach (undersampling + weights)
5. Robust evaluation metrics (C-index Uno's IPCW)

Based on research showing 55% error reduction with balanced approaches.

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
from sklearn.utils import resample
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
    cumulative_dynamic_auc
)
from sksurv.util import Surv
import joblib

logger = logging.getLogger(__name__)


class BalancedSurvivalTrainer:
    """
    Enhanced survival trainer with imbalance handling capabilities.

    Supports multiple strategies:
    - Standard training
    - Sample-weighted training
    - IPCW loss (Gradient Boosting)
    - Balanced RSF with undersampling
    - Hybrid approach
    """

    def __init__(self, random_state: int = 42, strategy: str = 'hybrid'):
        """
        Initialize balanced survival trainer.

        Args:
            random_state: Random seed
            strategy: Imbalance handling strategy
                     - 'standard': No special handling
                     - 'weighted': Sample weighting
                     - 'ipcw': IPCW loss for GBSA
                     - 'undersampled': Balanced RSF with undersampling
                     - 'hybrid': Undersampling + weights (RECOMMENDED)
        """
        self.random_state = random_state
        self.strategy = strategy
        self.models = {}
        self.feature_names = None
        self.training_results = {}
        self.sample_weights = None

        logger.info(f"BalancedSurvivalTrainer initialized:")
        logger.info(f"  - Strategy: {strategy}")
        logger.info(f"  - Random state: {random_state}")

    def calculate_sample_weights(
        self,
        y: np.ndarray,
        event_weight_ratio: float = 50.0
    ) -> np.ndarray:
        """
        Calculate sample weights based on event status.

        Args:
            y: Structured array with 'event' and 'duration' fields
            event_weight_ratio: Weight multiplier for events (default: 50)

        Returns:
            Array of sample weights
        """
        weights = np.ones(len(y))
        weights[y['event']] = event_weight_ratio

        n_events = y['event'].sum()
        n_censored = (~y['event']).sum()

        logger.info(f"Sample weights calculated:")
        logger.info(f"  - Events: {n_events:,} (weight={event_weight_ratio})")
        logger.info(f"  - Censored: {n_censored:,} (weight=1.0)")
        logger.info(f"  - Effective event proportion: {(n_events * event_weight_ratio) / (n_events * event_weight_ratio + n_censored):.1%}")

        return weights

    def undersample_survival_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersample censored cases to balance with events.

        Research shows 55% error reduction for Balanced RSF.

        Args:
            X: Features
            y: Structured array with 'event' and 'duration'
            ratio: Desired ratio of censored to events (default: 0.1 = 10:1)

        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        # Separate events and censored
        event_mask = y['event']
        censored_mask = ~event_mask

        X_events = X[event_mask]
        y_events = y[event_mask]
        X_censored = X[censored_mask]
        y_censored = y[censored_mask]

        n_events = len(y_events)
        n_censored_target = int(n_events / ratio)

        # Ensure we don't try to sample more than available
        n_censored_target = min(n_censored_target, len(y_censored))

        logger.info(f"Undersampling censored cases:")
        logger.info(f"  - Events: {n_events:,}")
        logger.info(f"  - Censored (original): {len(y_censored):,}")
        logger.info(f"  - Censored (target): {n_censored_target:,}")
        logger.info(f"  - Target ratio: {ratio:.1%} ({ratio:.0f}:1)")

        # Undersample censored cases
        X_censored_sampled, y_censored_sampled = resample(
            X_censored, y_censored,
            n_samples=n_censored_target,
            random_state=self.random_state,
            replace=False
        )

        # Combine
        X_balanced = np.vstack([X_events, X_censored_sampled])
        y_balanced = np.concatenate([y_events, y_censored_sampled])

        logger.info(f"  - Balanced dataset size: {len(y_balanced):,}")
        logger.info(f"  - Event rate after balancing: {y_balanced['event'].sum() / len(y_balanced) * 100:.1f}%")

        return X_balanced, y_balanced

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_event_col: str = 'event',
        target_duration_col: str = 'duration',
        test_size: float = 0.3
    ) -> Tuple:
        """
        Prepare data with appropriate balancing strategy.

        Returns:
            Different tuples based on strategy:
            - Standard/Weighted/IPCW: (X_train, X_test, y_train, y_test, feature_names, weights)
            - Undersampled: (X_train_bal, X_test, y_train_bal, y_test, feature_names, None)
            - Hybrid: (X_train_bal, X_test, y_train_bal, y_test, feature_names, weights)
        """
        logger.info(f"Preparing data with strategy: {self.strategy}")

        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = [
                target_event_col, target_duration_col,
                'geometry', 'fsq_place_id', 'name', 'address',
                'locality', 'region', 'postcode', 'country',
                'fsq_category_ids', 'fsq_category_labels',
                'primary_category', 'categories_list', 'category_count',
                'date_created', 'date_refreshed', 'date_closed',
                'date_created_parsed', 'date_closed_parsed',
                'categorical_label', 'poi_type', 'is_coffee_shop',
                'category_str', 'main_category'
            ]

            numeric_cols = df.select_dtypes(include=['number']).columns
            feature_cols = [
                col for col in numeric_cols
                if col not in exclude_cols and not col.endswith('_parsed')
            ]

            logger.info(f"Auto-detected {len(feature_cols)} numeric features")

        self.feature_names = feature_cols

        # Extract features
        X = df[feature_cols].values

        # Handle missing values
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
        logger.info(f"Event rate: {df[target_event_col].sum() / len(df) * 100:.2f}%")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df[target_event_col]
        )

        logger.info(f"Train set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")

        # Apply strategy-specific preprocessing
        if self.strategy == 'weighted':
            # Calculate weights for training set
            weights = self.calculate_sample_weights(y_train, event_weight_ratio=50.0)
            self.sample_weights = weights
            return X_train, X_test, y_train, y_test, feature_cols, weights

        elif self.strategy == 'undersampled':
            # Undersample training set
            X_train_balanced, y_train_balanced = self.undersample_survival_data(
                X_train, y_train, ratio=0.1  # 10:1 ratio
            )
            return X_train_balanced, X_test, y_train_balanced, y_test, feature_cols, None

        elif self.strategy == 'hybrid':
            # First undersample moderately
            X_train_under, y_train_under = self.undersample_survival_data(
                X_train, y_train, ratio=0.05  # 20:1 ratio (less aggressive)
            )
            # Then apply weights
            weights = self.calculate_sample_weights(y_train_under, event_weight_ratio=10.0)
            self.sample_weights = weights
            return X_train_under, X_test, y_train_under, y_test, feature_cols, weights

        else:  # 'standard' or 'ipcw'
            return X_train, X_test, y_train, y_test, feature_cols, None

    def train_random_survival_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_samples: float = 0.8,
        n_jobs: int = -1,
        **kwargs
    ) -> RandomSurvivalForest:
        """
        Train Random Survival Forest with imbalance handling.

        Adjusted parameters for imbalanced data:
        - Higher min_samples_split/leaf to prevent overfitting on rare events
        - More trees for stability
        - max_samples for additional regularization
        """
        logger.info(f"Training Random Survival Forest ({self.strategy} strategy)...")
        logger.info(f"  - n_estimators: {n_estimators}")
        logger.info(f"  - min_samples_split: {min_samples_split}")
        logger.info(f"  - min_samples_leaf: {min_samples_leaf}")
        logger.info(f"  - max_samples: {max_samples}")
        logger.info(f"  - Using sample weights: {sample_weight is not None}")

        model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=self.random_state,
            **kwargs
        )

        model.fit(X_train, y_train, sample_weight=sample_weight)

        self.models['random_survival_forest'] = model
        logger.info("Random Survival Forest training completed!")

        return model

    def train_gradient_boosting_survival(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_split: int = 20,
        min_samples_leaf: int = 50,
        subsample: float = 0.8,
        use_ipcw: bool = None,
        **kwargs
    ) -> GradientBoostingSurvivalAnalysis:
        """
        Train Gradient Boosting with imbalance handling.

        Auto-uses IPCW loss if strategy='ipcw', otherwise uses regular loss with weights.
        """
        # Auto-determine whether to use IPCW
        if use_ipcw is None:
            use_ipcw = (self.strategy == 'ipcw')

        loss = 'ipcwls' if use_ipcw else 'coxph'

        logger.info(f"Training Gradient Boosting Survival ({self.strategy} strategy)...")
        logger.info(f"  - Loss function: {loss}")
        logger.info(f"  - n_estimators: {n_estimators}")
        logger.info(f"  - learning_rate: {learning_rate}")
        logger.info(f"  - max_depth: {max_depth}")
        logger.info(f"  - subsample: {subsample}")
        logger.info(f"  - Using sample weights: {sample_weight is not None}")

        model = GradientBoostingSurvivalAnalysis(
            loss=loss,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=self.random_state,
            **kwargs
        )

        # Note: IPCW loss doesn't use sample_weight parameter
        if use_ipcw:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)

        self.models['gradient_boosting_survival'] = model
        logger.info("Gradient Boosting Survival training completed!")

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
        Evaluate model with ROBUST metrics for imbalanced data.

        Uses:
        - C-index (Harrell's) - standard
        - C-index (Uno's IPCW) - MORE ROBUST for imbalanced data
        - Integrated Brier Score
        - Time-dependent AUC
        """
        logger.info(f"\nEvaluating {model_name} (imbalance-robust metrics)...")

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        results = {'model_name': model_name, 'strategy': self.strategy}

        # Get risk scores
        risk_scores = model.predict(X_test)

        # 1. C-index (Harrell's) - Standard
        c_harrell = concordance_index_censored(
            y_test['event'],
            y_test['duration'],
            risk_scores
        )[0]
        results['c_index_harrell'] = float(c_harrell)

        logger.info(f"C-index (Harrell): {c_harrell:.4f}")

        # 2. C-index (Uno's IPCW) - MORE ROBUST for imbalanced data
        try:
            c_uno = concordance_index_ipcw(y_train, y_test, risk_scores)[0]
            results['c_index_uno'] = float(c_uno)
            logger.info(f"C-index (Uno IPCW): {c_uno:.4f} <- ROBUST metric")

            if c_uno > 0.8:
                logger.info("  -> Excellent discrimination!")
            elif c_uno > 0.7:
                logger.info("  -> Good discrimination")
            elif c_uno > 0.6:
                logger.info("  -> Fair discrimination")
            else:
                logger.info("  -> Poor discrimination")
        except Exception as e:
            logger.warning(f"Could not calculate Uno's C-index: {e}")
            results['c_index_uno'] = None

        # 3. Integrated Brier Score
        try:
            surv_funcs = model.predict_survival_function(X_test)
            times = np.array(time_points)

            from sksurv.metrics import brier_score
            brier_scores = []
            for i, t in enumerate(times):
                if (y_train['duration'] >= t).any():
                    bs = brier_score(y_train, y_test, [[fn(t) for fn in surv_funcs]], times=t)
                    brier_scores.append(bs[1][0])
                else:
                    brier_scores.append(np.nan)

            results['brier_scores'] = {
                f'{t}_days': float(bs) if not np.isnan(bs) else None
                for t, bs in zip(time_points, brier_scores)
            }

            valid_scores = [bs for bs in brier_scores if not np.isnan(bs)]
            if valid_scores:
                results['integrated_brier_score'] = float(np.mean(valid_scores))
                logger.info(f"Integrated Brier Score: {results['integrated_brier_score']:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate Brier Score: {e}")
            results['brier_scores'] = {}
            results['integrated_brier_score'] = None

        # 4. Time-dependent AUC
        try:
            auc_scores = {}
            for t in time_points:
                events_before_t = (y_test['duration'] <= t) & y_test['event']
                if events_before_t.sum() >= 5:
                    auc, mean_auc = cumulative_dynamic_auc(
                        y_train, y_test, risk_scores, times=[t]
                    )
                    auc_scores[f'{t}_days'] = float(auc[0])
                else:
                    auc_scores[f'{t}_days'] = None

            results['time_dependent_auc'] = auc_scores

            logger.info("Time-dependent AUC:")
            for time_label, auc_val in auc_scores.items():
                if auc_val is not None:
                    logger.info(f"  - {time_label}: {auc_val:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate time-dependent AUC: {e}")
            results['time_dependent_auc'] = {}

        results['time_points_evaluated'] = time_points

        self.training_results[model_name] = results

        return results

    def predict_survival_probability(
        self,
        model_name: str,
        X: np.ndarray,
        time_points: List[int] = [180, 365, 730, 1095]
    ) -> pd.DataFrame:
        """Predict survival probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        logger.info(f"Predicting survival probabilities ({model_name})...")

        surv_funcs = model.predict_survival_function(X)

        predictions = {}
        for t in time_points:
            prob_col = f'surv_prob_{t}d'
            predictions[prob_col] = [fn(t) for fn in surv_funcs]

        pred_df = pd.DataFrame(predictions)

        logger.info(f"Predictions shape: {pred_df.shape}")

        return pred_df

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        logger.info(f"\nTop {min(top_n, len(importance_df))} features ({model_name}):")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def save_model(self, model_name: str, output_dir: str = "models/survival_balanced"):
        """Save trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f"{model_name}_{self.strategy}.joblib"
        joblib.dump(self.models[model_name], model_file)
        logger.info(f"Model saved: {model_file}")

        # Save feature names
        features_file = output_path / f"{model_name}_{self.strategy}_features.json"
        with open(features_file, 'w') as f:
            json.dump({'feature_names': self.feature_names, 'strategy': self.strategy}, f, indent=2)
        logger.info(f"Features saved: {features_file}")

        # Save results
        if model_name in self.training_results:
            results_file = output_path / f"{model_name}_{self.strategy}_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.training_results[model_name], f, indent=2)
            logger.info(f"Results saved: {results_file}")

    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        if not self.training_results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.training_results.items():
            row = {
                'model': model_name,
                'strategy': results.get('strategy'),
                'c_index_harrell': results.get('c_index_harrell'),
                'c_index_uno': results.get('c_index_uno'),
                'integrated_brier_score': results.get('integrated_brier_score'),
            }

            if 'time_dependent_auc' in results:
                for time_label, auc in results['time_dependent_auc'].items():
                    row[f'auc_{time_label}'] = auc

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON (with imbalance handling)")
        logger.info("=" * 80)
        logger.info(comparison_df.to_string(index=False))
        logger.info("=" * 80)

        return comparison_df
