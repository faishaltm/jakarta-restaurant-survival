"""
Configuration loader utility
Loads and validates pipeline configuration from YAML
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigLoader:
    """Loads and validates pipeline configuration"""

    def __init__(self, config_path: str = None):
        """
        Initialize config loader

        Args:
            config_path: Path to config YAML file
        """
        if config_path is None:
            # Default to project root / config / pipeline_config.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "pipeline_config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        logger.info(f"Configuration loaded from: {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate required configuration keys"""
        required_sections = [
            'paths', 'geographic', 'poi_categories',
            'feature_engineering', 'model', 'evaluation'
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        logger.info("Configuration validated successfully")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path

        Args:
            key_path: Dot-separated path (e.g., 'geographic.bbox.min_lat')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_paths(self) -> Dict[str, Path]:
        """Get all paths as Path objects"""
        paths = self.config['paths'].copy()

        # Convert to Path objects
        project_root = Path(__file__).parent.parent.parent
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = project_root / value

        return paths

    def get_poi_filters(self) -> Dict[str, list]:
        """Get POI type filters"""
        return self.config['poi_categories']['poi_types']

    def get_coffee_keywords(self) -> list:
        """Get coffee shop keywords"""
        return self.config['poi_categories']['coffee_keywords']

    def get_buffer_distances(self) -> list:
        """Get buffer distances for feature engineering"""
        return self.config['feature_engineering']['buffer_distances_meters']

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific model

        Args:
            model_name: Model name (e.g., 'random_forest', 'xgboost')

        Returns:
            Model configuration dictionary
        """
        models = self.config['model']['models']
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        return models[model_name]

    def get_enabled_models(self) -> list:
        """Get list of enabled model names"""
        models = self.config['model']['models']
        return [name for name, cfg in models.items() if cfg.get('enabled', False)]

    def update(self, key_path: str, value: Any):
        """
        Update configuration value

        Args:
            key_path: Dot-separated path
            value: New value
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Updated config: {key_path} = {value}")

    def save(self, output_path: str = None):
        """
        Save current configuration to file

        Args:
            output_path: Path to save config (default: overwrite original)
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {output_path}")

    def __repr__(self):
        return f"ConfigLoader(path='{self.config_path}')"


if __name__ == "__main__":
    # Test config loader
    config = ConfigLoader()
    print(f"Project name: {config.get('project.name')}")
    print(f"Grid size: {config.get('geographic.grid_size_meters')}")
    print(f"Enabled models: {config.get_enabled_models()}")
    print(f"Buffer distances: {config.get_buffer_distances()}")
