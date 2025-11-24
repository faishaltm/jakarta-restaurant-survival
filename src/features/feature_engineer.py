"""
Feature Engineering Module
Creates spatial features from POI data based on literature best practices
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from shapely.geometry import Point
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import rasterio
from rasterio.mask import mask

from src.utils.config_loader import ConfigLoader


class FeatureEngineer:
    """Creates spatial features for coffee shop site selection"""

    def __init__(self, config: ConfigLoader):
        """
        Initialize feature engineer

        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.buffer_distances = config.get_buffer_distances()
        self.features_created = []

    def create_proximity_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        reference_gdfs: Dict[str, gpd.GeoDataFrame],
        suffix: str = ""
    ) -> gpd.GeoDataFrame:
        """
        Create proximity features (distance to nearest POI)

        Args:
            target_gdf: GeoDataFrame with target points (e.g., coffee shops or grid)
            reference_gdfs: Dict of {poi_type: GeoDataFrame} for reference POIs
            suffix: Suffix for column names

        Returns:
            target_gdf with distance columns added
        """
        logger.info(f"Creating proximity features for {len(target_gdf):,} targets...")

        # Must be in metric CRS
        if not target_gdf.crs.is_projected:
            logger.warning("Target GDF not in projected CRS, reprojecting...")
            utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
            target_gdf = target_gdf.to_crs(epsg=utm_epsg)

        for poi_type, ref_gdf in reference_gdfs.items():
            if len(ref_gdf) == 0:
                logger.warning(f"No {poi_type} POIs found, skipping")
                continue

            # Ensure same CRS
            if ref_gdf.crs != target_gdf.crs:
                ref_gdf = ref_gdf.to_crs(target_gdf.crs)

            # Calculate distance to nearest
            col_name = f"dist_nearest_{poi_type}{suffix}"
            target_gdf[col_name] = target_gdf.geometry.apply(
                lambda x: ref_gdf.distance(x).min()
            )

            self.features_created.append(col_name)
            logger.debug(f"Created feature: {col_name}")

        logger.success(f"Created {len(reference_gdfs)} proximity features")
        return target_gdf

    def create_density_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        reference_gdfs: Dict[str, gpd.GeoDataFrame],
        suffix: str = ""
    ) -> gpd.GeoDataFrame:
        """
        Create density features (count within buffer distances)

        Args:
            target_gdf: GeoDataFrame with target points
            reference_gdfs: Dict of {poi_type: GeoDataFrame}
            suffix: Suffix for column names

        Returns:
            target_gdf with count columns added
        """
        logger.info(f"Creating density features for {len(target_gdf):,} targets...")

        # Must be in metric CRS
        if not target_gdf.crs.is_projected:
            utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
            target_gdf = target_gdf.to_crs(epsg=utm_epsg)

        for poi_type, ref_gdf in reference_gdfs.items():
            if len(ref_gdf) == 0:
                logger.warning(f"No {poi_type} POIs found, skipping")
                continue

            # Ensure same CRS
            if ref_gdf.crs != target_gdf.crs:
                ref_gdf = ref_gdf.to_crs(target_gdf.crs)

            # Build spatial index for efficient querying
            ref_coords = np.array([[p.x, p.y] for p in ref_gdf.geometry])
            tree = cKDTree(ref_coords)

            target_coords = np.array([[p.x, p.y] for p in target_gdf.geometry])

            # Count within each buffer distance
            for buffer_m in self.buffer_distances:
                col_name = f"count_{poi_type}_{buffer_m}m{suffix}"

                # Query tree for points within buffer
                counts = tree.query_ball_point(target_coords, r=buffer_m, return_length=True)
                target_gdf[col_name] = counts

                self.features_created.append(col_name)
                logger.debug(f"Created feature: {col_name}")

        logger.success(
            f"Created density features for {len(reference_gdfs)} POI types "
            f"x {len(self.buffer_distances)} buffers"
        )
        return target_gdf

    def create_competitor_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        coffee_shops_gdf: gpd.GeoDataFrame,
        exclude_self: bool = True,
        suffix: str = ""
    ) -> gpd.GeoDataFrame:
        """
        Create competitor features (nearby coffee shops)

        Args:
            target_gdf: GeoDataFrame with target points
            coffee_shops_gdf: GeoDataFrame with all coffee shops
            exclude_self: If True, exclude target point from competitor count
            suffix: Suffix for column names

        Returns:
            target_gdf with competitor features
        """
        logger.info(f"Creating competitor features...")

        # Must be in metric CRS
        if not target_gdf.crs.is_projected:
            utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
            target_gdf = target_gdf.to_crs(epsg=utm_epsg)

        if coffee_shops_gdf.crs != target_gdf.crs:
            coffee_shops_gdf = coffee_shops_gdf.to_crs(target_gdf.crs)

        # Build spatial index
        coffee_coords = np.array([[p.x, p.y] for p in coffee_shops_gdf.geometry])
        tree = cKDTree(coffee_coords)

        target_coords = np.array([[p.x, p.y] for p in target_gdf.geometry])

        # Distance to nearest competitor
        distances, _ = tree.query(target_coords, k=2 if exclude_self else 1)
        if exclude_self:
            distances = distances[:, 1]  # Take second nearest (first is self)

        col_name_dist = f"dist_nearest_competitor{suffix}"
        target_gdf[col_name_dist] = distances
        self.features_created.append(col_name_dist)

        # Count competitors in buffers
        for buffer_m in self.buffer_distances:
            col_name = f"count_competitors_{buffer_m}m{suffix}"
            counts = tree.query_ball_point(target_coords, r=buffer_m, return_length=True)

            if exclude_self:
                counts = np.array(counts) - 1  # Subtract self
                counts = np.maximum(counts, 0)  # No negative counts

            target_gdf[col_name] = counts
            self.features_created.append(col_name)

        logger.success(f"Created competitor features")
        return target_gdf

    def create_diversity_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        all_pois_gdf: gpd.GeoDataFrame,
        suffix: str = ""
    ) -> gpd.GeoDataFrame:
        """
        Create POI diversity features (Shannon entropy, Simpson index)

        Args:
            target_gdf: GeoDataFrame with target points
            all_pois_gdf: GeoDataFrame with all categorized POIs
            suffix: Suffix for column names

        Returns:
            target_gdf with diversity features
        """
        logger.info("Creating diversity features...")

        if not target_gdf.crs.is_projected:
            utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
            target_gdf = target_gdf.to_crs(epsg=utm_epsg)

        if all_pois_gdf.crs != target_gdf.crs:
            all_pois_gdf = all_pois_gdf.to_crs(target_gdf.crs)

        # For each buffer distance
        for buffer_m in self.buffer_distances:
            shannon_col = f"poi_diversity_shannon_{buffer_m}m{suffix}"
            simpson_col = f"poi_diversity_simpson_{buffer_m}m{suffix}"

            shannon_values = []
            simpson_values = []

            for idx, target_point in target_gdf.iterrows():
                # Buffer around target
                buffer = target_point.geometry.buffer(buffer_m)

                # Find POIs within buffer
                nearby_pois = all_pois_gdf[all_pois_gdf.geometry.within(buffer)]

                if len(nearby_pois) == 0:
                    shannon_values.append(0)
                    simpson_values.append(0)
                    continue

                # Count by POI type
                type_counts = nearby_pois['poi_type'].value_counts()
                total = type_counts.sum()
                proportions = type_counts / total

                # Shannon entropy: -sum(p * log(p))
                shannon = -np.sum(proportions * np.log(proportions + 1e-10))
                shannon_values.append(shannon)

                # Simpson index: 1 - sum(p^2)
                simpson = 1 - np.sum(proportions ** 2)
                simpson_values.append(simpson)

            target_gdf[shannon_col] = shannon_values
            target_gdf[simpson_col] = simpson_values

            self.features_created.extend([shannon_col, simpson_col])

        logger.success("Created diversity features")
        return target_gdf

    def create_building_density_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        buildings_gdf: gpd.GeoDataFrame,
        suffix: str = ""
    ) -> gpd.GeoDataFrame:
        """
        Create building density features

        Args:
            target_gdf: GeoDataFrame with target points
            buildings_gdf: GeoDataFrame with building footprints
            suffix: Suffix for column names

        Returns:
            target_gdf with building density features
        """
        logger.info("Creating building density features...")

        if not target_gdf.crs.is_projected:
            utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
            target_gdf = target_gdf.to_crs(epsg=utm_epsg)

        if buildings_gdf.crs != target_gdf.crs:
            buildings_gdf = buildings_gdf.to_crs(target_gdf.crs)

        # Get building centroids for counting
        building_centroids = buildings_gdf.copy()
        building_centroids.geometry = building_centroids.geometry.centroid

        building_coords = np.array([[p.x, p.y] for p in building_centroids.geometry])
        tree = cKDTree(building_coords)

        target_coords = np.array([[p.x, p.y] for p in target_gdf.geometry])

        # Count buildings in buffers
        for buffer_m in self.buffer_distances:
            col_name = f"count_buildings_{buffer_m}m{suffix}"
            counts = tree.query_ball_point(target_coords, r=buffer_m, return_length=True)
            target_gdf[col_name] = counts
            self.features_created.append(col_name)

        logger.success("Created building density features")
        return target_gdf

    def extract_population_density(
        self,
        target_gdf: gpd.GeoDataFrame,
        population_raster: rasterio.DatasetReader,
        suffix: str = ""
    ) -> gpd.GeoDataFrame:
        """
        Extract population density from WorldPop raster

        Args:
            target_gdf: GeoDataFrame with target points
            population_raster: Rasterio dataset reader
            suffix: Suffix for column name

        Returns:
            target_gdf with population density column
        """
        logger.info("Extracting population density...")

        # Reproject to raster CRS
        target_gdf_reproj = target_gdf.to_crs(population_raster.crs)

        pop_values = []
        for idx, row in target_gdf_reproj.iterrows():
            try:
                # Sample raster at point location
                coords = [(row.geometry.x, row.geometry.y)]
                for val in population_raster.sample(coords):
                    pop_values.append(float(val[0]))
            except Exception as e:
                pop_values.append(np.nan)

        col_name = f"population_density{suffix}"
        target_gdf[col_name] = pop_values
        self.features_created.append(col_name)

        logger.success("Extracted population density")
        return target_gdf

    def create_all_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        data: Dict[str, gpd.GeoDataFrame],
        population_raster: Optional[rasterio.DatasetReader] = None
    ) -> gpd.GeoDataFrame:
        """
        Create all features in one go

        Args:
            target_gdf: GeoDataFrame with target points (coffee shops or grid)
            data: Dictionary with all loaded datasets
            population_raster: Optional population raster

        Returns:
            target_gdf with all features added
        """
        logger.info("=" * 60)
        logger.info("CREATING ALL FEATURES")
        logger.info("=" * 60)

        # Reproject to UTM
        utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
        target_gdf = target_gdf.to_crs(epsg=utm_epsg)

        # Proximity features (if enabled)
        if self.config.get('feature_engineering.proximity_features.enabled'):
            poi_types = self.config.get('feature_engineering.proximity_features.poi_types')
            reference_gdfs = {
                poi_type: data[f'poi_{poi_type}'].to_crs(epsg=utm_epsg)
                for poi_type in poi_types
                if f'poi_{poi_type}' in data
            }
            target_gdf = self.create_proximity_features(target_gdf, reference_gdfs)

        # Density features (if enabled)
        if self.config.get('feature_engineering.density_features.enabled'):
            poi_types = self.config.get('feature_engineering.density_features.poi_types')
            reference_gdfs = {
                poi_type: data[f'poi_{poi_type}'].to_crs(epsg=utm_epsg)
                for poi_type in poi_types
                if f'poi_{poi_type}' in data
            }
            target_gdf = self.create_density_features(target_gdf, reference_gdfs)

        # Competitor features (if enabled and coffee_shops available)
        if self.config.get('feature_engineering.proximity_features.include_competitors'):
            if 'coffee_shops' in data and len(data['coffee_shops']) > 0:
                coffee_shops = data['coffee_shops'].to_crs(epsg=utm_epsg)
                target_gdf = self.create_competitor_features(
                    target_gdf,
                    coffee_shops,
                    exclude_self=True
                )
            else:
                logger.info("Skipping competitor features (coffee_shops not available)")

        # Diversity features (if enabled)
        if self.config.get('feature_engineering.diversity_features.enabled'):
            # Use 'all_pois' or 'foursquare_all' whichever is available
            all_pois_key = 'all_pois' if 'all_pois' in data else 'foursquare_all'
            if all_pois_key in data and len(data[all_pois_key]) > 0:
                all_pois = data[all_pois_key].to_crs(epsg=utm_epsg)
                target_gdf = self.create_diversity_features(target_gdf, all_pois)
            else:
                logger.info("Skipping diversity features (all POIs data not available)")

        # Building density (if enabled)
        if self.config.get('feature_engineering.density_features.include_buildings'):
            buildings = data['buildings'].to_crs(epsg=utm_epsg)
            target_gdf = self.create_building_density_features(target_gdf, buildings)

        # Population density (if enabled and raster provided)
        if (self.config.get('feature_engineering.population_features.enabled')
            and population_raster is not None):
            target_gdf = self.extract_population_density(target_gdf, population_raster)

        logger.info("=" * 60)
        logger.success(f"Created {len(self.features_created)} features total")
        logger.info("=" * 60)

        return target_gdf

    def get_feature_names(self) -> List[str]:
        """Get list of created feature names"""
        return self.features_created

    def save_features(self, gdf: gpd.GeoDataFrame, output_path: str):
        """
        Save features to file

        Args:
            gdf: GeoDataFrame with features
            output_path: Path to save
        """
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.csv':
            gdf.drop('geometry', axis=1).to_csv(output_path, index=False)
        elif output_path.suffix == '.geojson':
            gdf.to_file(output_path, driver='GeoJSON')
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")

        logger.success(f"Features saved to: {output_path}")


if __name__ == "__main__":
    # Test feature engineering
    from loguru import logger
    import sys
    from src.data.data_loader import DataLoader

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = ConfigLoader()
    loader = DataLoader(config)
    engineer = FeatureEngineer(config)

    # Load sample data
    data = loader.load_all(sample_foursquare=10000)

    # Create features for coffee shops
    coffee_with_features = engineer.create_all_features(
        data['coffee_shops'].head(100),
        data
    )

    print("\n=== Features Created ===")
    print(f"Total features: {len(engineer.get_feature_names())}")
    print(f"\nSample features:\n{coffee_with_features.columns.tolist()}")
