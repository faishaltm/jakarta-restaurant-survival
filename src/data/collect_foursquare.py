"""
Foursquare Open Source Places Data Collection

Downloads and processes Foursquare's 8M+ free Indonesian POIs.

Data source: https://opensource.foursquare.com/os-places/
License: Apache 2.0 (free for commercial use)

This provides high-quality POI data including:
- Name, address, coordinates
- Categories (fsq_category_id)
- Place IDs (fsq_id)
"""

import os
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional
from shapely.geometry import Point, box
from loguru import logger

# Jakarta bounding box
JAKARTA_BBOX = {
    'min_lon': 106.6,
    'min_lat': -6.4,
    'max_lon': 107.1,
    'max_lat': -6.0
}

# Foursquare Open Source download URL
# Note: This is an example URL structure - actual URL may vary
# Visit https://opensource.foursquare.com/os-places/ for current download links
FOURSQUARE_OS_DOWNLOAD_PAGE = "https://opensource.foursquare.com/os-places/"

class FoursquareCollector:
    """Collects Foursquare Open Source Places data for Jakarta"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "foursquare"
        self.processed_dir = self.data_dir / "processed" / "foursquare"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_indonesia_dataset(self, output_path: Optional[Path] = None) -> Path:
        """
        Download Foursquare OS Places for Indonesia

        Note: Due to the nature of Foursquare OS data distribution,
        users need to manually download from the website.

        Args:
            output_path: Path to save the downloaded file

        Returns:
            Path to downloaded file
        """
        if output_path is None:
            output_path = self.raw_dir / "foursquare_indonesia.parquet"

        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path

        logger.warning("Manual download required!")
        logger.info("Please follow these steps:")
        logger.info("1. Visit: " + FOURSQUARE_OS_DOWNLOAD_PAGE)
        logger.info("2. Find Indonesia dataset (or Asia/Southeast Asia region)")
        logger.info("3. Download the file (usually .parquet or .csv format)")
        logger.info(f"4. Save to: {output_path}")
        logger.info("5. Run this script again")

        raise FileNotFoundError(f"Please manually download Indonesia dataset to {output_path}")

    def load_foursquare_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load Foursquare OS Places data from file

        Args:
            file_path: Path to Foursquare data file (.parquet or .csv)

        Returns:
            DataFrame with Foursquare POI data
        """
        logger.info(f"Loading Foursquare data from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect file format and load accordingly
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.success(f"Loaded {len(df):,} POIs")
        logger.info(f"Columns: {df.columns.tolist()}")

        return df

    def filter_jakarta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter POIs to Jakarta region only

        Args:
            df: DataFrame with Foursquare POI data (must have latitude/longitude columns)

        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering POIs for Jakarta region...")

        # Check for coordinate columns (various possible names)
        lat_col = None
        lon_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower:
                lat_col = col
            if 'lon' in col_lower or 'lng' in col_lower:
                lon_col = col

        if not lat_col or not lon_col:
            logger.error(f"Could not find latitude/longitude columns in: {df.columns.tolist()}")
            raise ValueError("DataFrame must have latitude and longitude columns")

        logger.info(f"Using columns: {lat_col}, {lon_col}")

        # Filter by bounding box
        mask = (
            (df[lat_col] >= JAKARTA_BBOX['min_lat']) &
            (df[lat_col] <= JAKARTA_BBOX['max_lat']) &
            (df[lon_col] >= JAKARTA_BBOX['min_lon']) &
            (df[lon_col] <= JAKARTA_BBOX['max_lon'])
        )

        jakarta_df = df[mask].copy()

        logger.success(f"Filtered to {len(jakarta_df):,} POIs in Jakarta ({len(jakarta_df)/len(df)*100:.1f}%)")

        return jakarta_df

    def convert_to_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Convert DataFrame to GeoDataFrame with geometry

        Args:
            df: DataFrame with latitude/longitude columns

        Returns:
            GeoDataFrame
        """
        logger.info("Converting to GeoDataFrame...")

        # Find coordinate columns
        lat_col = None
        lon_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower:
                lat_col = col
            if 'lon' in col_lower or 'lng' in col_lower:
                lon_col = col

        if not lat_col or not lon_col:
            raise ValueError("DataFrame must have latitude and longitude columns")

        # Create geometry column
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=geometry,
            crs="EPSG:4326"
        )

        logger.success(f"Created GeoDataFrame with {len(gdf)} POIs")

        return gdf

    def filter_coffee_shops(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter for coffee shops and cafes specifically

        Args:
            gdf: GeoDataFrame with Foursquare POIs

        Returns:
            Filtered GeoDataFrame
        """
        logger.info("Filtering for coffee shops and cafes...")

        # Check for category column
        category_col = None
        for col in gdf.columns:
            if 'category' in col.lower() or 'fsq_category' in col.lower():
                category_col = col
                break

        if not category_col:
            logger.warning("No category column found, cannot filter by category")
            return gdf

        # Filter for coffee/cafe related categories
        # Category IDs from Foursquare taxonomy
        coffee_keywords = [
            'coffee', 'cafe', 'café', 'kopi', 'kedai kopi',
            'coffeehouse', 'espresso'
        ]

        mask = gdf[category_col].str.lower().str.contains(
            '|'.join(coffee_keywords),
            na=False,
            case=False
        )

        coffee_gdf = gdf[mask].copy()

        logger.success(f"Found {len(coffee_gdf)} coffee shops/cafes")

        return coffee_gdf

    def process_and_save(self, input_file: Path) -> dict:
        """
        Complete processing pipeline: load, filter, and save

        Args:
            input_file: Path to raw Foursquare data file

        Returns:
            Dictionary with paths to processed files
        """
        logger.info("Processing Foursquare data...")

        # Load data
        df = self.load_foursquare_data(input_file)

        # Filter to Jakarta
        jakarta_df = self.filter_jakarta(df)

        # Convert to GeoDataFrame
        gdf = self.convert_to_geodataframe(jakarta_df)

        # Save all POIs
        all_pois_path = self.processed_dir / "jakarta_pois_foursquare.geojson"
        gdf.to_file(all_pois_path, driver='GeoJSON')
        logger.success(f"Saved all Jakarta POIs to: {all_pois_path}")

        # Also save as CSV for easier viewing
        csv_path = self.processed_dir / "jakarta_pois_foursquare.csv"
        jakarta_df.to_csv(csv_path, index=False)
        logger.success(f"Saved CSV to: {csv_path}")

        # Filter and save coffee shops specifically
        coffee_gdf = self.filter_coffee_shops(gdf)

        if len(coffee_gdf) > 0:
            coffee_path = self.processed_dir / "jakarta_coffee_shops_foursquare.geojson"
            coffee_gdf.to_file(coffee_path, driver='GeoJSON')
            logger.success(f"Saved coffee shops to: {coffee_path}")

            coffee_csv_path = self.processed_dir / "jakarta_coffee_shops_foursquare.csv"
            coffee_gdf.drop(columns='geometry').to_csv(coffee_csv_path, index=False)
            logger.success(f"Saved coffee shops CSV to: {coffee_csv_path}")

            results = {
                'all_pois_geojson': all_pois_path,
                'all_pois_csv': csv_path,
                'coffee_shops_geojson': coffee_path,
                'coffee_shops_csv': coffee_csv_path
            }
        else:
            results = {
                'all_pois_geojson': all_pois_path,
                'all_pois_csv': csv_path
            }

        return results

    def collect_all(self) -> dict:
        """
        Run complete Foursquare data collection and processing

        Returns:
            Dictionary with paths to processed files
        """
        logger.info("Starting Foursquare data collection for Jakarta...")

        # Check if raw data file exists
        possible_files = [
            self.raw_dir / "foursquare_indonesia.parquet",
            self.raw_dir / "foursquare_indonesia.csv",
            self.raw_dir / "indonesia.parquet",
            self.raw_dir / "indonesia.csv"
        ]

        input_file = None
        for file_path in possible_files:
            if file_path.exists():
                input_file = file_path
                logger.info(f"Found input file: {input_file}")
                break

        if input_file is None:
            # Attempt download (will provide manual instructions)
            self.download_indonesia_dataset()
            return {}

        # Process the data
        results = self.process_and_save(input_file)

        logger.success("Foursquare data collection complete!")
        logger.info(f"Results: {results}")

        return results


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect Foursquare OS Places data for Jakarta')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--input-file', help='Path to raw Foursquare data file')

    args = parser.parse_args()

    collector = FoursquareCollector(data_dir=args.data_dir)

    if args.input_file:
        input_path = Path(args.input_file)
        results = collector.process_and_save(input_path)
    else:
        results = collector.collect_all()

    if results:
        print("\n" + "="*60)
        print("Foursquare Data Collection Complete!")
        print("="*60)
        for key, path in results.items():
            print(f"{key}: {path}")
    else:
        print("\nPlease download the Foursquare Indonesia dataset manually")
        print("Visit: " + FOURSQUARE_OS_DOWNLOAD_PAGE)


if __name__ == "__main__":
    main()
