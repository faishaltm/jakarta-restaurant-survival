"""
Data Loader Module
Handles loading and preprocessing of all data sources
"""
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger
import ast

from src.utils.config_loader import ConfigLoader


class DataLoader:
    """Loads and preprocesses all data sources"""

    def __init__(self, config: ConfigLoader):
        """
        Initialize data loader

        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.paths = config.get_paths()

    def load_foursquare_pois(self, sample_size: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Load Foursquare POI data

        Args:
            sample_size: If provided, randomly sample this many POIs

        Returns:
            GeoDataFrame with Foursquare POIs
        """
        logger.info("Loading Foursquare POI data...")

        # Optimized dtypes for memory efficiency
        dtypes = {
            'fsq_place_id': 'string',
            'name': 'string',
            'latitude': 'float32',
            'longitude': 'float32',
            'address': 'string',
            'locality': 'string',
            'region': 'string',
            'postcode': 'string',
            'country': 'string'
        }

        df = pd.read_csv(self.paths['foursquare_csv'], dtype=dtypes)

        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size:,} POIs from {len(df):,}")
            df = df.sample(sample_size, random_state=self.config.get('project.random_seed'))

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
            crs=f"EPSG:{self.config.get('geographic.wgs84_epsg')}"
        )

        # Parse categories
        gdf = self._parse_categories(gdf)

        logger.success(f"Loaded {len(gdf):,} Foursquare POIs")
        return gdf

    def _parse_categories(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Parse category arrays from string representation"""

        def parse_category_array(cat_str):
            if pd.isna(cat_str):
                return []
            try:
                if isinstance(cat_str, list):
                    return cat_str
                return ast.literal_eval(cat_str)
            except:
                return []

        gdf['categories_list'] = gdf['fsq_category_labels'].apply(parse_category_array)
        gdf['primary_category'] = gdf['categories_list'].apply(
            lambda x: x[0] if len(x) > 0 else 'Unknown'
        )
        gdf['category_count'] = gdf['categories_list'].apply(len)

        return gdf

    def load_osm_pois(self) -> gpd.GeoDataFrame:
        """Load OpenStreetMap POI data"""
        logger.info("Loading OSM POI data...")
        gdf = gpd.read_file(self.paths['osm_geojson'])
        logger.success(f"Loaded {len(gdf):,} OSM POIs")
        return gdf

    def load_buildings(self) -> gpd.GeoDataFrame:
        """Load building footprints"""
        logger.info("Loading building data...")
        gdf = gpd.read_file(self.paths['buildings_geojson'])
        logger.success(f"Loaded {len(gdf):,} buildings")
        return gdf

    def load_boundaries(self, filter_jakarta: bool = True) -> gpd.GeoDataFrame:
        """
        Load administrative boundaries

        Args:
            filter_jakarta: If True, filter to Jakarta only

        Returns:
            GeoDataFrame with boundaries
        """
        logger.info("Loading administrative boundaries...")
        gdf = gpd.read_file(self.paths['boundaries_geojson'])

        if filter_jakarta:
            jakarta_keywords = ['jakarta', 'dki jakarta', 'kepulauan seribu']
            gdf = gdf[
                gdf['NAME_1'].str.lower().str.contains('|'.join(jakarta_keywords), na=False)
            ].copy()
            logger.success(f"Loaded {len(gdf):,} Jakarta districts")
        else:
            logger.success(f"Loaded {len(gdf):,} districts")

        return gdf

    def load_population_raster(self) -> rasterio.DatasetReader:
        """
        Load population density raster

        Returns:
            Rasterio dataset reader
        """
        logger.info("Loading population density raster...")
        raster = rasterio.open(self.paths['population_tif'])
        logger.success(f"Loaded population raster: {raster.shape}")
        return raster

    def filter_coffee_shops(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter coffee shops from POI data

        Args:
            gdf: GeoDataFrame with POIs

        Returns:
            GeoDataFrame with coffee shops only
        """
        logger.info("Filtering coffee shops...")
        keywords = self.config.get_coffee_keywords()

        def is_coffee_shop(categories_list):
            if not categories_list:
                return False

            for cat in categories_list:
                cat_lower = str(cat).lower()
                for keyword in keywords:
                    if keyword in cat_lower:
                        return True
            return False

        gdf['is_coffee_shop'] = gdf['categories_list'].apply(is_coffee_shop)
        gdf_coffee = gdf[gdf['is_coffee_shop']].copy()

        logger.success(f"Found {len(gdf_coffee):,} coffee shops ({len(gdf_coffee)/len(gdf)*100:.2f}%)")
        return gdf_coffee

    def categorize_pois(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Categorize POIs by type (university, office, mall, etc.)

        Args:
            gdf: GeoDataFrame with POIs

        Returns:
            GeoDataFrame with poi_type column
        """
        logger.info("Categorizing POIs by type...")
        poi_filters = self.config.get_poi_filters()

        def categorize_poi(categories_list):
            if not categories_list:
                return 'other'

            categories_str = ' '.join([str(cat).lower() for cat in categories_list])

            for poi_type, keywords in poi_filters.items():
                for keyword in keywords:
                    if keyword in categories_str:
                        return poi_type

            return 'other'

        gdf['poi_type'] = gdf['categories_list'].apply(categorize_poi)

        # Log distribution
        poi_counts = gdf['poi_type'].value_counts()
        logger.info(f"POI type distribution:\n{poi_counts.head(10)}")

        return gdf

    def extract_poi_by_type(
        self,
        gdf: gpd.GeoDataFrame,
        poi_type: str
    ) -> gpd.GeoDataFrame:
        """
        Extract POIs of specific type

        Args:
            gdf: GeoDataFrame with categorized POIs
            poi_type: Type to extract (e.g., 'university', 'office')

        Returns:
            GeoDataFrame with POIs of specified type
        """
        gdf_filtered = gdf[gdf['poi_type'] == poi_type].copy()
        logger.info(f"Extracted {len(gdf_filtered):,} {poi_type} POIs")
        return gdf_filtered

    def reproject_to_utm(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Reproject GeoDataFrame to UTM (for metric calculations)

        Args:
            gdf: GeoDataFrame in WGS84

        Returns:
            GeoDataFrame in UTM
        """
        utm_epsg = self.config.get('geographic.utm_jakarta_epsg')
        return gdf.to_crs(epsg=utm_epsg)

    def load_all(self, sample_foursquare: Optional[int] = None) -> Dict[str, gpd.GeoDataFrame]:
        """
        Load all datasets at once

        Args:
            sample_foursquare: If provided, sample Foursquare data

        Returns:
            Dictionary with all loaded datasets
        """
        logger.info("Loading all datasets...")

        data = {}

        # Load Foursquare
        gdf_fsq = self.load_foursquare_pois(sample_size=sample_foursquare)
        gdf_fsq = self.categorize_pois(gdf_fsq)
        data['foursquare_all'] = gdf_fsq

        # Filter coffee shops
        data['coffee_shops'] = self.filter_coffee_shops(gdf_fsq)

        # Extract key POI types
        poi_types = list(self.config.get_poi_filters().keys())
        for poi_type in poi_types:
            data[f'poi_{poi_type}'] = self.extract_poi_by_type(gdf_fsq, poi_type)

        # Load other datasets
        data['osm_pois'] = self.load_osm_pois()
        data['buildings'] = self.load_buildings()
        data['boundaries'] = self.load_boundaries()

        logger.success(f"Loaded {len(data)} datasets")
        return data


if __name__ == "__main__":
    # Test data loader
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = ConfigLoader()
    loader = DataLoader(config)

    # Test loading (sample for speed)
    data = loader.load_all(sample_foursquare=10000)

    print("\n=== Loaded Datasets ===")
    for name, gdf in data.items():
        if isinstance(gdf, gpd.GeoDataFrame):
            print(f"{name}: {len(gdf):,} records")
