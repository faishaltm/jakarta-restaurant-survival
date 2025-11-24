"""
Coffee Shop Location Collection for Training Data

Collects locations of successful coffee shop chains in Jakarta:
- Kopi Kenangan (1,000+ outlets)
- Janji Jiwa (900+ outlets)
- Fore Coffee (700+ outlets)
- Starbucks, other major chains

Uses legitimate APIs:
- Google Places API (preferred, $200 free monthly credit)
- Foursquare Places API (10,000 free calls)
- Previously collected Foursquare OS data

This data serves as positive training examples (successful locations)
"""

import os
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List, Dict, Optional
from shapely.geometry import Point
from loguru import logger
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")

# Jakarta bounding box
JAKARTA_BBOX = {
    'min_lon': 106.6,
    'min_lat': -6.4,
    'max_lon': 107.1,
    'max_lat': -6.0,
    'center_lat': -6.2088,
    'center_lon': 106.8456
}

# Indonesian coffee shop brands to search for
COFFEE_BRANDS = [
    "Kopi Kenangan",
    "Janji Jiwa",
    "Fore Coffee",
    "Starbucks",
    "Kopi Tuku",
    "Anomali Coffee",
    "Filosofi Kopi",
    "Tanamera Coffee",
    "Excelso",
    "The Coffee Bean"
]


class CoffeeShopCollector:
    """Collects coffee shop locations for training data"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "coffee_shops"
        self.processed_dir = self.data_dir / "processed" / "coffee_shops"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.google_api_key = GOOGLE_API_KEY
        self.foursquare_api_key = FOURSQUARE_API_KEY

    def search_google_places(self, query: str, location: tuple = None) -> List[Dict]:
        """
        Search for places using Google Places API Text Search

        Args:
            query: Search query (e.g., "Kopi Kenangan Jakarta")
            location: (lat, lon) tuple for center point

        Returns:
            List of place dictionaries
        """
        if not self.google_api_key:
            logger.warning("Google Places API key not found")
            return []

        if location is None:
            location = (JAKARTA_BBOX['center_lat'], JAKARTA_BBOX['center_lon'])

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

        params = {
            'query': query,
            'location': f"{location[0]},{location[1]}",
            'radius': 25000,  # 25km radius to cover Jakarta
            'key': self.google_api_key
        }

        results = []

        try:
            logger.info(f"Searching Google Places for: {query}")

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'OK':
                places = data.get('results', [])
                results.extend(places)
                logger.success(f"Found {len(places)} places")

                # Handle pagination (up to 60 results total)
                next_page_token = data.get('next_page_token')

                while next_page_token and len(results) < 60:
                    time.sleep(2)  # Required delay between pagination requests

                    page_params = {
                        'pagetoken': next_page_token,
                        'key': self.google_api_key
                    }

                    response = requests.get(url, params=page_params, timeout=30)
                    response.raise_for_status()

                    data = response.json()

                    if data.get('status') == 'OK':
                        places = data.get('results', [])
                        results.extend(places)
                        logger.info(f"Fetched next page: {len(places)} places")

                    next_page_token = data.get('next_page_token')

            elif data.get('status') == 'ZERO_RESULTS':
                logger.warning(f"No results found for: {query}")
            else:
                logger.error(f"Google Places API error: {data.get('status')}")
                if 'error_message' in data:
                    logger.error(f"Error message: {data['error_message']}")

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Google Places API request failed: {e}")
            return []

    def search_foursquare_places(self, query: str, location: tuple = None) -> List[Dict]:
        """
        Search for places using Foursquare Places API

        Args:
            query: Search query (e.g., "Kopi Kenangan")
            location: (lat, lon) tuple for center point

        Returns:
            List of place dictionaries
        """
        if not self.foursquare_api_key:
            logger.warning("Foursquare API key not found")
            return []

        if location is None:
            location = (JAKARTA_BBOX['center_lat'], JAKARTA_BBOX['center_lon'])

        url = "https://api.foursquare.com/v3/places/search"

        headers = {
            "Authorization": self.foursquare_api_key,
            "Accept": "application/json"
        }

        params = {
            'query': query,
            'll': f"{location[0]},{location[1]}",
            'radius': 25000,  # 25km
            'limit': 50
        }

        try:
            logger.info(f"Searching Foursquare for: {query}")

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = data.get('results', [])

            logger.success(f"Found {len(results)} places")

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Foursquare API request failed: {e}")
            return []

    def parse_google_place(self, place: Dict) -> Dict:
        """Parse Google Places API result to standard format"""
        return {
            'source': 'google',
            'place_id': place.get('place_id'),
            'name': place.get('name'),
            'address': place.get('formatted_address'),
            'latitude': place.get('geometry', {}).get('location', {}).get('lat'),
            'longitude': place.get('geometry', {}).get('location', {}).get('lng'),
            'rating': place.get('rating'),
            'user_ratings_total': place.get('user_ratings_total'),
            'types': ','.join(place.get('types', [])),
            'business_status': place.get('business_status')
        }

    def parse_foursquare_place(self, place: Dict) -> Dict:
        """Parse Foursquare API result to standard format"""
        return {
            'source': 'foursquare',
            'place_id': place.get('fsq_id'),
            'name': place.get('name'),
            'address': place.get('location', {}).get('formatted_address'),
            'latitude': place.get('geocodes', {}).get('main', {}).get('latitude'),
            'longitude': place.get('geocodes', {}).get('main', {}).get('longitude'),
            'rating': place.get('rating'),
            'user_ratings_total': place.get('stats', {}).get('total_ratings'),
            'types': ','.join([cat.get('name', '') for cat in place.get('categories', [])]),
            'business_status': None
        }

    def collect_brand_locations(self, brand_name: str, use_google: bool = True,
                                use_foursquare: bool = True) -> pd.DataFrame:
        """
        Collect all locations for a specific brand

        Args:
            brand_name: Name of the coffee shop brand
            use_google: Whether to search Google Places API
            use_foursquare: Whether to search Foursquare API

        Returns:
            DataFrame with all locations found
        """
        logger.info(f"Collecting locations for: {brand_name}")

        all_places = []

        # Search Google Places
        if use_google and self.google_api_key:
            google_results = self.search_google_places(f"{brand_name} Jakarta")
            all_places.extend([self.parse_google_place(p) for p in google_results])
            time.sleep(1)  # Rate limiting

        # Search Foursquare
        if use_foursquare and self.foursquare_api_key:
            fsq_results = self.search_foursquare_places(brand_name)
            all_places.extend([self.parse_foursquare_place(p) for p in fsq_results])
            time.sleep(1)  # Rate limiting

        if not all_places:
            logger.warning(f"No locations found for {brand_name}")
            return pd.DataFrame()

        df = pd.DataFrame(all_places)
        df['brand'] = brand_name

        # Remove duplicates based on coordinates (within 50 meters)
        df = df.dropna(subset=['latitude', 'longitude'])
        df = self._remove_duplicate_locations(df)

        logger.success(f"Collected {len(df)} unique locations for {brand_name}")

        return df

    def _remove_duplicate_locations(self, df: pd.DataFrame, threshold_meters: float = 50) -> pd.DataFrame:
        """
        Remove duplicate locations that are very close to each other

        Args:
            df: DataFrame with latitude/longitude columns
            threshold_meters: Distance threshold in meters

        Returns:
            Deduplicated DataFrame
        """
        from scipy.spatial.distance import cdist
        import numpy as np

        if len(df) == 0:
            return df

        # Convert lat/lon to approximate meters (rough approximation for Jakarta)
        # 1 degree lat ~ 111km, 1 degree lon ~ 111km at equator
        coords = df[['latitude', 'longitude']].values
        coords_km = coords * [111, 111]  # Convert to km

        # Calculate pairwise distances
        distances = cdist(coords_km, coords_km, metric='euclidean')

        # Find duplicates (distance < threshold)
        threshold_km = threshold_meters / 1000
        duplicates = np.triu(distances < threshold_km, k=1)

        # Keep first occurrence, remove duplicates
        to_keep = []
        removed = set()

        for i in range(len(df)):
            if i not in removed:
                to_keep.append(i)
                # Mark all nearby points as removed
                nearby = np.where(duplicates[i])[0]
                removed.update(nearby)

        deduplicated = df.iloc[to_keep].copy()

        if len(df) > len(deduplicated):
            logger.info(f"Removed {len(df) - len(deduplicated)} duplicate locations")

        return deduplicated

    def collect_all_brands(self, brands: List[str] = None) -> gpd.GeoDataFrame:
        """
        Collect locations for all coffee shop brands

        Args:
            brands: List of brand names (defaults to COFFEE_BRANDS)

        Returns:
            GeoDataFrame with all collected locations
        """
        if brands is None:
            brands = COFFEE_BRANDS

        logger.info(f"Collecting locations for {len(brands)} brands...")

        all_data = []

        for brand in brands:
            try:
                brand_df = self.collect_brand_locations(brand)
                if len(brand_df) > 0:
                    all_data.append(brand_df)
            except Exception as e:
                logger.error(f"Failed to collect {brand}: {e}")

        if not all_data:
            logger.error("No coffee shop data collected!")
            return gpd.GeoDataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        logger.success(f"Total locations collected: {len(combined_df)}")

        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(combined_df['longitude'], combined_df['latitude'])]
        gdf = gpd.GeoDataFrame(combined_df, geometry=geometry, crs="EPSG:4326")

        return gdf

    def collect_all(self) -> dict:
        """
        Run complete coffee shop data collection

        Returns:
            Dictionary with paths to collected data
        """
        logger.info("Starting coffee shop data collection...")

        # Check API keys
        if not self.google_api_key and not self.foursquare_api_key:
            logger.error("No API keys found!")
            logger.info("Please set GOOGLE_PLACES_API_KEY or FOURSQUARE_API_KEY in .env file")
            return {}

        # Collect data
        gdf = self.collect_all_brands()

        if len(gdf) == 0:
            logger.error("No data collected")
            return {}

        # Save to files
        geojson_path = self.processed_dir / "jakarta_coffee_shops_training.geojson"
        gdf.to_file(geojson_path, driver='GeoJSON')
        logger.success(f"Saved GeoJSON to: {geojson_path}")

        csv_path = self.processed_dir / "jakarta_coffee_shops_training.csv"
        gdf.drop(columns='geometry').to_csv(csv_path, index=False)
        logger.success(f"Saved CSV to: {csv_path}")

        # Print summary statistics
        print("\n" + "="*60)
        print("Coffee Shop Collection Summary")
        print("="*60)
        print(f"Total locations: {len(gdf)}")
        print("\nBy brand:")
        print(gdf['brand'].value_counts())
        print("\nBy source:")
        print(gdf['source'].value_counts())

        results = {
            'geojson': geojson_path,
            'csv': csv_path,
            'total_locations': len(gdf)
        }

        return results


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect coffee shop locations for training')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--brand', help='Collect only specific brand')

    args = parser.parse_args()

    collector = CoffeeShopCollector(data_dir=args.data_dir)

    if args.brand:
        df = collector.collect_brand_locations(args.brand)
        if len(df) > 0:
            print(f"\nCollected {len(df)} locations for {args.brand}")
            print(df.head())
    else:
        results = collector.collect_all()

        if results:
            print("\n" + "="*60)
            print("Collection Complete!")
            print("="*60)
            for key, value in results.items():
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
