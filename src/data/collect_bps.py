"""
BPS (Badan Pusat Statistik) API Integration

Collects demographic and economic data from Indonesian Statistics Bureau.

API Documentation: https://webapi.bps.go.id/developer/
Free API key required (register at link above)

Data collected:
- Population by kelurahan/kecamatan
- Income levels
- Employment statistics
- Business establishments
"""

import os
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BPS API Configuration
BPS_BASE_URL = "https://webapi.bps.go.id/v1/api"
BPS_API_KEY = os.getenv("BPS_API_KEY")

# Jakarta province code in BPS system
JAKARTA_PROVINCE_CODE = "31"  # DKI Jakarta

class BPSCollector:
    """Collects demographic data from BPS API"""

    def __init__(self, api_key: Optional[str] = None, data_dir: str = "./data"):
        self.api_key = api_key or BPS_API_KEY
        if not self.api_key:
            logger.warning("BPS API key not found. Set BPS_API_KEY environment variable.")
            logger.info("Get free API key from: https://webapi.bps.go.id/developer/")

        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "bps"
        self.processed_dir = self.data_dir / "processed" / "bps"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make request to BPS API

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        if not self.api_key:
            raise ValueError("BPS API key is required. Set BPS_API_KEY environment variable.")

        url = f"{BPS_BASE_URL}/{endpoint}"
        params['key'] = self.api_key

        try:
            logger.debug(f"Requesting: {url}")
            logger.debug(f"Params: {params}")

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"BPS API request failed: {e}")
            raise

    def get_provinces(self) -> pd.DataFrame:
        """
        Get list of all provinces

        Returns:
            DataFrame with province codes and names
        """
        logger.info("Fetching provinces from BPS...")

        try:
            data = self._make_request("domain", {"type": "prov"})

            if 'data' in data and isinstance(data['data'], list):
                df = pd.DataFrame(data['data'][1])  # Index 1 contains the data
                logger.success(f"Retrieved {len(df)} provinces")
                return df
            else:
                logger.error(f"Unexpected response format: {data}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get provinces: {e}")
            raise

    def get_regencies(self, province_code: str = JAKARTA_PROVINCE_CODE) -> pd.DataFrame:
        """
        Get list of regencies/cities in a province

        Args:
            province_code: BPS province code (default: Jakarta)

        Returns:
            DataFrame with regency codes and names
        """
        logger.info(f"Fetching regencies for province {province_code}...")

        try:
            data = self._make_request("domain", {
                "type": "kab",
                "prov": province_code
            })

            if 'data' in data and isinstance(data['data'], list):
                df = pd.DataFrame(data['data'][1])
                logger.success(f"Retrieved {len(df)} regencies")
                return df
            else:
                logger.error(f"Unexpected response format: {data}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get regencies: {e}")
            raise

    def get_available_indicators(self, province_code: str = JAKARTA_PROVINCE_CODE) -> pd.DataFrame:
        """
        Get list of available statistical indicators

        Args:
            province_code: BPS province code

        Returns:
            DataFrame with indicator IDs and descriptions
        """
        logger.info(f"Fetching available indicators for province {province_code}...")

        try:
            data = self._make_request("list", {
                "model": "data",
                "domain": province_code,
                "lang": "eng"
            })

            if 'data' in data and isinstance(data['data'], list):
                indicators = []
                for item in data['data'][1]:
                    if isinstance(item, dict):
                        indicators.append(item)

                df = pd.DataFrame(indicators)
                logger.success(f"Retrieved {len(df)} indicators")
                return df
            else:
                logger.error(f"Unexpected response format: {data}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get indicators: {e}")
            raise

    def get_indicator_data(self, indicator_id: str, province_code: str = JAKARTA_PROVINCE_CODE) -> pd.DataFrame:
        """
        Get data for a specific indicator

        Args:
            indicator_id: BPS indicator ID
            province_code: BPS province code

        Returns:
            DataFrame with indicator data
        """
        logger.info(f"Fetching data for indicator {indicator_id}...")

        try:
            data = self._make_request("list", {
                "model": "data",
                "domain": province_code,
                "var": indicator_id,
                "lang": "eng"
            })

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                logger.success(f"Retrieved {len(df)} records")
                return df
            else:
                logger.error(f"Unexpected response format: {data}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get indicator data: {e}")
            raise

    def get_jakarta_demographics(self) -> pd.DataFrame:
        """
        Get key demographic indicators for Jakarta

        Returns:
            DataFrame with demographic data by region
        """
        logger.info("Collecting Jakarta demographic data...")

        # Get administrative divisions first
        regencies = self.get_regencies(JAKARTA_PROVINCE_CODE)

        # Get available indicators
        indicators = self.get_available_indicators(JAKARTA_PROVINCE_CODE)

        logger.info(f"Found {len(indicators)} available indicators")

        # Save indicator list for reference
        indicators_path = self.raw_dir / "jakarta_indicators.csv"
        indicators.to_csv(indicators_path, index=False)
        logger.info(f"Saved indicator list to: {indicators_path}")

        # Try to collect key demographic indicators
        # Note: Actual indicator IDs may vary, need to check the indicator list
        demographic_data = {
            'regency': regencies.to_dict('records') if not regencies.empty else []
        }

        # Save raw data
        output_path = self.raw_dir / "jakarta_demographics.json"
        import json
        with open(output_path, 'w') as f:
            json.dump(demographic_data, f, indent=2)

        logger.success(f"Saved demographic data to: {output_path}")

        return pd.DataFrame(demographic_data)

    def collect_all(self) -> dict:
        """
        Run complete BPS data collection pipeline for Jakarta

        Returns:
            Dictionary with paths to collected data
        """
        logger.info("Starting BPS data collection for Jakarta...")

        results = {}

        try:
            # Get provinces list
            provinces = self.get_provinces()
            provinces_path = self.processed_dir / "provinces.csv"
            provinces.to_csv(provinces_path, index=False)
            results['provinces'] = provinces_path
            logger.success(f"Saved provinces to: {provinces_path}")

            # Get Jakarta regencies
            regencies = self.get_regencies(JAKARTA_PROVINCE_CODE)
            regencies_path = self.processed_dir / "jakarta_regencies.csv"
            regencies.to_csv(regencies_path, index=False)
            results['regencies'] = regencies_path
            logger.success(f"Saved Jakarta regencies to: {regencies_path}")

            # Get available indicators
            indicators = self.get_available_indicators(JAKARTA_PROVINCE_CODE)
            indicators_path = self.processed_dir / "jakarta_indicators.csv"
            indicators.to_csv(indicators_path, index=False)
            results['indicators'] = indicators_path
            logger.success(f"Saved indicators to: {indicators_path}")

            # Get demographic data
            demographics = self.get_jakarta_demographics()
            results['demographics'] = self.raw_dir / "jakarta_demographics.json"

            logger.success("BPS data collection complete!")
            logger.info(f"Results: {results}")

            return results

        except Exception as e:
            logger.error(f"BPS data collection failed: {e}")
            raise


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect BPS demographic data for Jakarta')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--api-key', help='BPS API key (or set BPS_API_KEY env var)')

    args = parser.parse_args()

    if args.api_key:
        os.environ['BPS_API_KEY'] = args.api_key

    collector = BPSCollector(data_dir=args.data_dir)

    if not collector.api_key:
        print("\n" + "="*60)
        print("BPS API KEY REQUIRED")
        print("="*60)
        print("Get a free API key from: https://webapi.bps.go.id/developer/")
        print("Then set it as environment variable: BPS_API_KEY=your_key")
        print("Or pass it with --api-key parameter")
        return

    results = collector.collect_all()

    print("\n" + "="*60)
    print("BPS Data Collection Complete!")
    print("="*60)
    for key, path in results.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
