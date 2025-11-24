"""
Coffee Shop Site Selection - ML Model Training

Trains Random Forest classifier to predict successful coffee shop locations.

Uses spatial cross-validation to prevent overfitting due to spatial autocorrelation.
Target: Binary classification (successful location vs random location)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from loguru import logger
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from features.spatial_features import SpatialFeatureEngineer


class CoffeeShopModel:
    """Coffee shop site selection ML model"""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 10,
        random_state: int = 42
    ):
        """
        Initialize model

        Args:
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split node
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced classes
        )

        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def prepare_training_data(
        self,
        positive_samples: gpd.GeoDataFrame,
        negative_samples: gpd.GeoDataFrame,
        feature_matrix: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with labels

        Args:
            positive_samples: Known successful locations
            negative_samples: Random/unsuccessful locations
            feature_matrix: Engineered features

        Returns:
            X, y arrays
        """
        logger.info("Preparing training data...")

        # Create labels
        y_positive = np.ones(len(positive_samples))
        y_negative = np.zeros(len(negative_samples))
        y = np.concatenate([y_positive, y_negative])

        logger.info(f"Positive samples: {len(positive_samples)}")
        logger.info(f"Negative samples: {len(negative_samples)}")
        logger.info(f"Class balance: {y.mean():.2%} positive")

        # Get features
        X = feature_matrix.values
        self.feature_names = feature_matrix.columns.tolist()

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features: {self.feature_names}")

        return X, y

    def spatial_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: gpd.GeoDataFrame,
        n_folds: int = 5
    ) -> Dict:
        """
        Perform spatial cross-validation

        Splits data by geographic clusters to prevent spatial leakage

        Args:
            X: Feature matrix
            y: Labels
            locations: GeoDataFrame with geometries
            n_folds: Number of CV folds

        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Performing {n_folds}-fold spatial cross-validation...")

        from sklearn.cluster import KMeans

        # Cluster locations geographically
        coords = np.array([[geom.x, geom.y] for geom in locations.geometry])
        kmeans = KMeans(n_clusters=n_folds, random_state=42)
        spatial_folds = kmeans.fit_predict(coords)

        logger.info(f"Created {n_folds} geographic clusters for CV")

        # Manual CV with spatial folds
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }

        for fold in range(n_folds):
            # Split by spatial cluster
            test_mask = spatial_folds == fold
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred))
            scores['recall'].append(recall_score(y_test, y_pred))
            scores['f1'].append(f1_score(y_test, y_pred))
            scores['auc'].append(roc_auc_score(y_test, y_pred_proba))

            logger.info(f"Fold {fold + 1}: Accuracy={scores['accuracy'][-1]:.3f}, AUC={scores['auc'][-1]:.3f}")

        # Calculate mean and std
        cv_results = {}
        for metric, values in scores.items():
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)

        logger.success("Spatial cross-validation complete")
        logger.info(f"Mean accuracy: {cv_results['accuracy_mean']:.3f} (+/- {cv_results['accuracy_std']:.3f})")
        logger.info(f"Mean AUC: {cv_results['auc_mean']:.3f} (+/- {cv_results['auc_std']:.3f})")

        return cv_results

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Train final model on all data

        Args:
            X: Feature matrix
            y: Labels
        """
        logger.info("Training final model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)

        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.success("Model training complete")
        logger.info("\nTop 10 most important features:")
        print(self.feature_importance.head(10))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score_location(self, X: np.ndarray) -> pd.DataFrame:
        """
        Score locations with suitability scores (0-100)

        Args:
            X: Feature matrix

        Returns:
            DataFrame with scores and probabilities
        """
        probabilities = self.predict_proba(X)[:, 1]

        # Convert to 0-100 score
        scores = probabilities * 100

        # Calculate percentiles
        percentiles = pd.Series(scores).rank(pct=True) * 100

        results = pd.DataFrame({
            'probability': probabilities,
            'score': scores,
            'percentile': percentiles
        })

        return results

    def save_model(self, output_path: str) -> None:
        """Save model to disk"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, output_path)
        logger.success(f"Model saved to: {output_path}")

    @classmethod
    def load_model(cls, model_path: str) -> 'CoffeeShopModel':
        """Load model from disk"""
        model_data = joblib.load(model_path)

        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.feature_importance = model_data['feature_importance']

        logger.success(f"Model loaded from: {model_path}")
        return instance


def generate_negative_samples(
    positive_samples: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    n_samples: int = None,
    min_distance: float = 200
) -> gpd.GeoDataFrame:
    """
    Generate negative training samples (random locations)

    Args:
        positive_samples: Known successful locations
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        n_samples: Number of negative samples (default: 2x positive samples)
        min_distance: Minimum distance from positive samples (meters)

    Returns:
        GeoDataFrame with negative samples
    """
    if n_samples is None:
        n_samples = len(positive_samples) * 2

    logger.info(f"Generating {n_samples} negative samples...")

    from shapely.geometry import Point

    # Project positive samples for distance calculation
    positive_proj = positive_samples.to_crs("EPSG:32748")

    negative_points = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(negative_points) < n_samples and attempts < max_attempts:
        # Generate random point
        lon = np.random.uniform(bbox[0], bbox[2])
        lat = np.random.uniform(bbox[1], bbox[3])
        point = Point(lon, lat)

        # Check distance from positive samples
        point_proj = gpd.GeoDataFrame(
            [1],
            geometry=[point],
            crs="EPSG:4326"
        ).to_crs("EPSG:32748")

        distances = positive_proj.geometry.distance(point_proj.geometry.iloc[0])

        if distances.min() > min_distance:
            negative_points.append(point)

        attempts += 1

    logger.success(f"Generated {len(negative_points)} negative samples")

    # Create GeoDataFrame
    negative_gdf = gpd.GeoDataFrame(
        {'id': range(len(negative_points))},
        geometry=negative_points,
        crs="EPSG:4326"
    )

    return negative_gdf


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Train coffee shop site selection model')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--output-dir', default='./models', help='Model output directory')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Coffee Shop Site Selection - Model Training")
    logger.info("="*60)

    # This is a template - actual training would require loading real data
    logger.info("To train the model:")
    logger.info("1. Load positive samples (known coffee shop locations)")
    logger.info("2. Generate negative samples (random locations)")
    logger.info("3. Engineer features using SpatialFeatureEngineer")
    logger.info("4. Train model with spatial cross-validation")
    logger.info("5. Save trained model")
    logger.info("\nSee notebooks/04_model_training.ipynb for complete example")


if __name__ == "__main__":
    main()
