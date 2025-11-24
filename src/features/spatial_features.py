"""
Spatial Feature Engineering

Calculates spatial features for location intelligence:
- Competitor density in multiple buffers (500m, 1km, 2km)
- POI diversity indices
- Distance to nearest competitors
- Same-brand cannibalization metrics
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Tuple
from loguru import logger
from scipy.stats import entropy


class SpatialFeatureEngineer:
    """Calculate spatial features for ML model"""

    def __init__(self, buffer_distances: List[int] = None):
        """
        Initialize feature engineer

        Args:
            buffer_distances: List of buffer distances in meters (default: [500, 1000, 2000])
        """
        self.buffer_distances = buffer_distances or [500, 1000, 2000]

    def calculate_competitor_density(
        self,
        locations: gpd.GeoDataFrame,
        competitors: gpd.GeoDataFrame,
        buffer_distance: int
    ) -> pd.Series:
        """
        Calculate number of competitors within buffer distance

        Args:
            locations: GeoDataFrame of candidate locations
            competitors: GeoDataFrame of existing competitor locations
            buffer_distance: Buffer radius in meters

        Returns:
            Series with competitor counts
        """
        logger.info(f"Calculating competitor density ({buffer_distance}m buffers)...")

        # Create buffers (convert to projected CRS for accurate meters)
        locations_proj = locations.to_crs("EPSG:32748")  # UTM Zone 48S for Jakarta
        competitors_proj = competitors.to_crs("EPSG:32748")

        buffers = locations_proj.geometry.buffer(buffer_distance)

        # Count competitors within each buffer
        counts = []
        for buffer in buffers:
            within = competitors_proj.geometry.within(buffer)
            counts.append(within.sum())

        return pd.Series(counts, index=locations.index)

    def calculate_distance_to_nearest(
        self,
        locations: gpd.GeoDataFrame,
        targets: gpd.GeoDataFrame
    ) -> pd.Series:
        """
        Calculate distance to nearest target location

        Args:
            locations: GeoDataFrame of candidate locations
            targets: GeoDataFrame of target locations (e.g., competitors)

        Returns:
            Series with distances in meters
        """
        logger.info("Calculating distance to nearest target...")

        if len(targets) == 0:
            return pd.Series([np.inf] * len(locations), index=locations.index)

        # Project to UTM for accurate meters
        locations_proj = locations.to_crs("EPSG:32748")
        targets_proj = targets.to_crs("EPSG:32748")

        distances = []
        for loc_geom in locations_proj.geometry:
            dists = targets_proj.geometry.distance(loc_geom)
            distances.append(dists.min() if len(dists) > 0 else np.inf)

        return pd.Series(distances, index=locations.index)

    def calculate_poi_diversity(
        self,
        locations: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        buffer_distance: int,
        category_column: str = 'category'
    ) -> pd.Series:
        """
        Calculate POI diversity using Shannon entropy

        Args:
            locations: GeoDataFrame of candidate locations
            pois: GeoDataFrame of all POIs with category information
            buffer_distance: Buffer radius in meters
            category_column: Column name containing POI categories

        Returns:
            Series with diversity indices (0 = no diversity, higher = more diverse)
        """
        logger.info(f"Calculating POI diversity ({buffer_distance}m buffers)...")

        if category_column not in pois.columns:
            logger.warning(f"Category column '{category_column}' not found in POIs")
            return pd.Series([0] * len(locations), index=locations.index)

        # Project to UTM
        locations_proj = locations.to_crs("EPSG:32748")
        pois_proj = pois.to_crs("EPSG:32748")

        buffers = locations_proj.geometry.buffer(buffer_distance)

        diversities = []
        for buffer in buffers:
            # Get POIs within buffer
            within_mask = pois_proj.geometry.within(buffer)
            pois_within = pois_proj[within_mask]

            if len(pois_within) == 0:
                diversities.append(0)
                continue

            # Calculate category distribution
            category_counts = pois_within[category_column].value_counts()
            proportions = category_counts / category_counts.sum()

            # Calculate Shannon entropy
            diversity = entropy(proportions)
            diversities.append(diversity)

        return pd.Series(diversities, index=locations.index)

    def calculate_same_brand_cannibalization(
        self,
        locations: gpd.GeoDataFrame,
        same_brand_locations: gpd.GeoDataFrame,
        buffer_distance: int = 2000
    ) -> pd.Series:
        """
        Calculate number of same-brand locations within buffer (cannibalization risk)

        Args:
            locations: GeoDataFrame of candidate locations
            same_brand_locations: GeoDataFrame of existing same-brand locations
            buffer_distance: Buffer radius in meters (default 2km)

        Returns:
            Series with same-brand counts
        """
        logger.info(f"Calculating same-brand cannibalization ({buffer_distance}m)...")

        return self.calculate_competitor_density(
            locations,
            same_brand_locations,
            buffer_distance
        )

    def calculate_all_buffer_features(
        self,
        locations: gpd.GeoDataFrame,
        competitors: gpd.GeoDataFrame,
        all_pois: gpd.GeoDataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate all buffer-based spatial features

        Args:
            locations: GeoDataFrame of candidate locations
            competitors: GeoDataFrame of competitor locations
            all_pois: GeoDataFrame of all POIs (for diversity calculation)

        Returns:
            DataFrame with all spatial features
        """
        logger.info("Calculating all spatial features...")

        features = pd.DataFrame(index=locations.index)

        # Competitor density at multiple buffer sizes
        for buffer_dist in self.buffer_distances:
            col_name = f'competitor_count_{buffer_dist}m'
            features[col_name] = self.calculate_competitor_density(
                locations,
                competitors,
                buffer_dist
            )

        # Distance to nearest competitor
        features['nearest_competitor_dist'] = self.calculate_distance_to_nearest(
            locations,
            competitors
        )

        # POI diversity (if all_pois provided)
        if all_pois is not None and len(all_pois) > 0:
            for buffer_dist in [500, 1000]:
                col_name = f'poi_diversity_{buffer_dist}m'
                features[col_name] = self.calculate_poi_diversity(
                    locations,
                    all_pois,
                    buffer_dist
                )

        logger.success(f"Calculated {len(features.columns)} spatial features")

        return features

    def calculate_population_in_buffers(
        self,
        locations: gpd.GeoDataFrame,
        population_raster_path: str = None
    ) -> pd.DataFrame:
        """
        Calculate population within buffer zones using raster data

        Args:
            locations: GeoDataFrame of candidate locations
            population_raster_path: Path to WorldPop or similar population raster

        Returns:
            DataFrame with population counts per buffer
        """
        logger.info("Calculating population in buffers...")

        if population_raster_path is None:
            logger.warning("No population raster provided, skipping population features")
            return pd.DataFrame(index=locations.index)

        try:
            import rasterio
            from rasterio.mask import mask

            features = pd.DataFrame(index=locations.index)

            # Project locations
            locations_proj = locations.to_crs("EPSG:32748")

            with rasterio.open(population_raster_path) as src:
                for buffer_dist in self.buffer_distances:
                    col_name = f'population_{buffer_dist}m'
                    populations = []

                    for geom in locations_proj.geometry:
                        buffer = geom.buffer(buffer_dist)

                        # Reproject buffer to raster CRS
                        buffer_gdf = gpd.GeoDataFrame(
                            [1],
                            geometry=[buffer],
                            crs="EPSG:32748"
                        )
                        buffer_reproj = buffer_gdf.to_crs(src.crs)

                        # Extract raster values within buffer
                        try:
                            out_image, out_transform = mask(
                                src,
                                buffer_reproj.geometry,
                                crop=True
                            )
                            population = out_image[out_image > 0].sum()
                            populations.append(population)
                        except Exception:
                            populations.append(0)

                    features[col_name] = populations

            return features

        except ImportError:
            logger.warning("rasterio not available, skipping population features")
            return pd.DataFrame(index=locations.index)
        except Exception as e:
            logger.error(f"Population calculation failed: {e}")
            return pd.DataFrame(index=locations.index)


def create_feature_matrix(
    locations: gpd.GeoDataFrame,
    competitors: gpd.GeoDataFrame,
    all_pois: gpd.GeoDataFrame = None,
    demographics: gpd.GeoDataFrame = None,
    population_raster: str = None
) -> pd.DataFrame:
    """
    Create complete feature matrix for ML model

    Args:
        locations: Candidate locations to score
        competitors: Existing competitor locations
        all_pois: All POIs for diversity calculation
        demographics: Demographic data by region
        population_raster: Path to population density raster

    Returns:
        DataFrame with all features
    """
    logger.info("Creating complete feature matrix...")

    engineer = SpatialFeatureEngineer()

    # Calculate spatial features
    spatial_features = engineer.calculate_all_buffer_features(
        locations,
        competitors,
        all_pois
    )

    # Calculate population features (if raster provided)
    if population_raster:
        pop_features = engineer.calculate_population_in_buffers(
            locations,
            population_raster
        )
        spatial_features = pd.concat([spatial_features, pop_features], axis=1)

    # Add demographic features (if provided)
    if demographics is not None:
        # Spatial join to get demographics for each location
        locations_with_demo = gpd.sjoin(
            locations,
            demographics,
            how='left',
            predicate='within'
        )

        demo_cols = [col for col in demographics.columns if col != 'geometry']
        for col in demo_cols:
            if col in locations_with_demo.columns:
                spatial_features[col] = locations_with_demo[col].values

    logger.success(f"Feature matrix created: {spatial_features.shape}")

    return spatial_features


if __name__ == "__main__":
    # Example usage
    logger.info("Spatial Feature Engineering Module")
    logger.info("This module provides spatial feature calculation for ML model")
    logger.info("Import and use SpatialFeatureEngineer class in your notebooks/scripts")
