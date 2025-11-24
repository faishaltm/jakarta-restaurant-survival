"""
Model Training Module
Handles model training, hyperparameter tuning, and feature selection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

from src.utils.config_loader import ConfigLoader


class ModelTrainer:
    """Trains and tunes ML models for coffee shop site selection"""

    def __init__(self, config: ConfigLoader):
        """
        Initialize model trainer

        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.random_seed = config.get('project.random_seed')
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'label',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training

        Args:
            df: DataFrame with features and labels
            target_col: Name of target column
            feature_cols: List of feature columns (if None, use all except target)

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing training data...")

        # Handle missing values
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values, filling with median...")
            df = df.fillna(df.median(numeric_only=True))

        # Separate features and target
        if feature_cols is None:
            # Exclude non-numeric columns and target
            exclude_cols = [target_col, 'geometry', 'fsq_place_id', 'name', 'address',
                           'locality', 'region', 'postcode', 'country', 'primary_category',
                           'categories_list', 'category_count', 'poi_type', 'is_coffee_shop',
                           'date_created', 'date_refreshed', 'date_closed']

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        self.feature_names = feature_cols

        X = df[feature_cols]
        y = df[target_col]

        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Samples: {len(df):,} ({y.value_counts().to_dict()})")

        # Train-test split
        test_size = self.config.get('model.test_size')
        stratify = y if self.config.get('model.stratify') else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify
        )

        logger.success(
            f"Split data: train={len(X_train):,}, test={len(X_test):,}"
        )

        return X_train, X_test, y_train, y_test

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None
    ) -> RandomForestClassifier:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameters: Hyperparameters to use (if None, use config defaults)

        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest...")

        if hyperparameters is None:
            # Use first value from config as default
            hp_config = self.config.get_model_config('random_forest')['hyperparameters']
            hyperparameters = {
                'n_estimators': hp_config['n_estimators'][0],
                'max_depth': hp_config['max_depth'][0],
                'min_samples_split': hp_config['min_samples_split'][0],
                'min_samples_leaf': hp_config['min_samples_leaf'][0],
                'max_features': hp_config['max_features'][0],
                'random_state': self.random_seed
            }

        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        logger.success(f"Random Forest trained with {hyperparameters}")
        return model

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameters: Hyperparameters to use

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost...")

        if hyperparameters is None:
            hp_config = self.config.get_model_config('xgboost')['hyperparameters']
            hyperparameters = {
                'n_estimators': hp_config['n_estimators'][0],
                'max_depth': hp_config['max_depth'][0],
                'learning_rate': hp_config['learning_rate'][0],
                'subsample': hp_config['subsample'][0],
                'colsample_bytree': hp_config['colsample_bytree'][0],
                'random_state': self.random_seed
            }

        model = xgb.XGBClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        logger.success(f"XGBoost trained with {hyperparameters}")
        return model

    def tune_hyperparameters(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[Any, Dict]:
        """
        Tune hyperparameters using grid/random search

        Args:
            model_name: Name of model ('random_forest', 'xgboost')
            X_train: Training features
            y_train: Training labels

        Returns:
            (best_model, best_params)
        """
        logger.info(f"Tuning hyperparameters for {model_name}...")

        model_config = self.config.get_model_config(model_name)
        param_grid = model_config['hyperparameters']

        # Initialize base model
        if model_name == 'random_forest':
            base_model = RandomForestClassifier(random_state=self.random_seed)
        elif model_name == 'xgboost':
            base_model = xgb.XGBClassifier(random_state=self.random_seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Setup search
        tuning_config = self.config.get('model.tuning')
        method = tuning_config['method']
        cv_folds = tuning_config['cv_folds']
        scoring = tuning_config['scoring']
        n_jobs = tuning_config['n_jobs']

        if method == 'grid_search':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        elif method == 'random_search':
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_seed,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")

        # Fit
        logger.info(f"Running {method} with {cv_folds}-fold CV...")
        search.fit(X_train, y_train)

        logger.success(f"Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")

        return search.best_estimator_, search.best_params_

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        tune: bool = True
    ) -> Dict[str, Any]:
        """
        Train all enabled models

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            tune: Whether to tune hyperparameters

        Returns:
            Dictionary with trained models and scores
        """
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)

        results = {}
        enabled_models = self.config.get_enabled_models()

        for model_name in enabled_models:
            logger.info(f"\n--- Training {model_name} ---")

            if tune:
                model, best_params = self.tune_hyperparameters(
                    model_name, X_train, y_train
                )
            else:
                if model_name == 'random_forest':
                    model = self.train_random_forest(X_train, y_train)
                elif model_name == 'xgboost':
                    model = self.train_xgboost(X_train, y_train)
                else:
                    logger.warning(f"Unknown model: {model_name}, skipping")
                    continue

                best_params = None

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.success(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            results[model_name] = {
                'model': model,
                'best_params': best_params,
                'accuracy': accuracy,
                'f1': f1,
                'predictions': y_pred
            }

            self.models[model_name] = model

        # Select best model
        best_name = max(results, key=lambda x: results[x]['f1'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name

        logger.info("=" * 60)
        logger.success(f"Best model: {best_name} (F1: {results[best_name]['f1']:.4f})")
        logger.info("=" * 60)

        return results

    def get_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from model

        Args:
            model_name: Model name (if None, use best model)
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]

        # Get importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} doesn't have feature_importances_")
            return pd.DataFrame()

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} features for {model_name}:")
        logger.info(f"\n{importance_df.head(top_n)}")

        return importance_df

    def save_model(
        self,
        model_name: Optional[str] = None,
        output_dir: str = "models"
    ):
        """
        Save trained model

        Args:
            model_name: Model to save (if None, save best model)
            output_dir: Directory to save model
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f"{model_name}_model.pkl"
        joblib.dump(model, model_file)
        logger.success(f"Model saved: {model_file}")

        # Save feature names
        features_file = output_path / f"{model_name}_features.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.success(f"Features saved: {features_file}")

    def load_model(
        self,
        model_name: str,
        model_dir: str = "models"
    ) -> Any:
        """
        Load saved model

        Args:
            model_name: Name of model to load
            model_dir: Directory where model is saved

        Returns:
            Loaded model
        """
        model_path = Path(model_dir) / f"{model_name}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        logger.success(f"Model loaded: {model_path}")

        # Load feature names
        features_path = Path(model_dir) / f"{model_name}_features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)

        return model


if __name__ == "__main__":
    # Test model trainer
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name='label')

    df = pd.concat([X, y], axis=1)

    # Test training
    config = ConfigLoader()
    trainer = ModelTrainer(config)

    X_train, X_test, y_train, y_test = trainer.prepare_training_data(df)

    # Train without tuning (faster)
    results = trainer.train_all_models(
        X_train, X_test, y_train, y_test,
        tune=False
    )

    # Feature importance
    importance = trainer.get_feature_importance()
