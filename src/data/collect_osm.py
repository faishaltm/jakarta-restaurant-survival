"""
OpenStreetMap Data Collection for Jakarta

Downloads and processes OSM data for Jakarta region including:
- POIs (Points of Interest)
- Road network
- Building footprints
- Amenities

Data source: Geofabrik (https://download.geofabrik.de/asia/indonesia.html)
"""

import os
import urllib.request
from pathlib import Path
from typing import Tuple
import geopandas as gpd
from shapely.geometry import box
from loguru import logger

# Jakarta bounding box (rough approximation)
JAKARTA_BBOX = {
    'min_lon': 106.6,
    'min_lat': -6.4,
    'max_lon': 107.1,
    'max_lat': -6.0
}

# Geofabrik Indonesia extract URL
OSM_URL = "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf"

class OSMCollector:
    """Collects and processes OpenStreetMap data for Jakarta"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "osm"
        self.processed_dir = self.data_dir / "processed" / "osm"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_jakarta_bbox(self) -> box:
        """Get Jakarta bounding box as Shapely geometry"""
        return box(
            JAKARTA_BBOX['min_lon'],
            JAKARTA_BBOX['min_lat'],
            JAKARTA_BBOX['max_lon'],
            JAKARTA_BBOX['max_lat']
        )

    def download_indonesia_extract(self) -> Path:
        """
        Download Indonesia OSM extract from Geofabrik

        Returns:
            Path to downloaded file
        """
        output_path = self.raw_dir / "indonesia-latest.osm.pbf"

        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path

        logger.info(f"Downloading OSM Indonesia extract from Geofabrik...")
        logger.info(f"URL: {OSM_URL}")
        logger.info(f"Output: {output_path}")
        logger.warning("This is a large file (~500-800MB), download may take 10-30 minutes")

        try:
            # Download with progress reporting
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 100 == 0:  # Report every 100 blocks
                    logger.info(f"Downloaded: {downloaded / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB ({percent:.1f}%)")

            urllib.request.urlretrieve(OSM_URL, output_path, reporthook=report_progress)
            logger.success(f"Download complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def extract_jakarta_pois_osmium(self, osm_file: Path) -> Path:
        """
        Extract Jakarta POIs from OSM PBF using osmium-tool

        This requires osmium-tool to be installed:
        - Windows: Download from https://osmcode.org/osmium-tool/
        - Linux: apt-get install osmium-tool
        - Mac: brew install osmium-tool

        Args:
            osm_file: Path to Indonesia OSM PBF file

        Returns:
            Path to extracted Jakarta OSM file
        """
        import subprocess

        output_path = self.raw_dir / "jakarta.osm.pbf"

        if output_path.exists():
            logger.info(f"Jakarta extract already exists: {output_path}")
            return output_path

        bbox_str = f"{JAKARTA_BBOX['min_lon']},{JAKARTA_BBOX['min_lat']},{JAKARTA_BBOX['max_lon']},{JAKARTA_BBOX['max_lat']}"

        logger.info(f"Extracting Jakarta region using osmium...")
        logger.info(f"Bounding box: {bbox_str}")

        try:
            cmd = [
                "osmium",
                "extract",
                "--bbox", bbox_str,
                "--output", str(output_path),
                str(osm_file)
            ]

            subprocess.run(cmd, check=True)
            logger.success(f"Jakarta extract complete: {output_path}")
            return output_path

        except FileNotFoundError:
            logger.error("osmium-tool not found!")
            logger.info("Please install osmium-tool:")
            logger.info("  Windows: https://osmcode.org/osmium-tool/")
            logger.info("  Linux: apt-get install osmium-tool")
            logger.info("  Mac: brew install osmium-tool")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"osmium extraction failed: {e}")
            raise

    def extract_pois_with_osmnx(self) -> gpd.GeoDataFrame:
        """
        Extract POIs for Jakarta using OSMnx (simpler, no osmium required)

        This is slower but doesn't require additional tools.

        Returns:
            GeoDataFrame with POIs
        """
        import osmnx as ox

        logger.info("Extracting Jakarta POIs using OSMnx...")

        # Define tags for coffee shops and related amenities
        tags = {
            'amenity': ['cafe', 'restaurant', 'fast_food', 'bar'],
            'shop': ['coffee', 'convenience', 'supermarket'],
            'tourism': ['attraction', 'hotel']
        }

        try:
            # Download POIs within Jakarta bounding box
            bbox = (JAKARTA_BBOX['max_lat'], JAKARTA_BBOX['min_lat'],
                   JAKARTA_BBOX['max_lon'], JAKARTA_BBOX['min_lon'])  # north, south, east, west

            logger.info(f"Downloading POIs for bbox: {bbox}")
            pois = ox.features_from_bbox(bbox=bbox, tags=tags)

            logger.success(f"Downloaded {len(pois)} POIs")

            # Save to file
            output_path = self.processed_dir / "jakarta_pois_osm.geojson"
            pois.to_file(output_path, driver='GeoJSON')
            logger.success(f"Saved POIs to: {output_path}")

            return pois

        except Exception as e:
            logger.error(f"OSMnx POI extraction failed: {e}")
            raise

    def extract_road_network(self) -> gpd.GeoDataFrame:
        """
        Extract road network for Jakarta using OSMnx

        Returns:
            GeoDataFrame with road network
        """
        import osmnx as ox

        logger.info("Extracting Jakarta road network using OSMnx...")

        try:
            # Download road network
            bbox = (JAKARTA_BBOX['max_lat'], JAKARTA_BBOX['min_lat'],
                   JAKARTA_BBOX['max_lon'], JAKARTA_BBOX['min_lon'])

            logger.info("Downloading drive network...")
            G = ox.graph_from_bbox(
                bbox=bbox,
                network_type='drive',
                simplify=True
            )

            # Convert to GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)

            logger.success(f"Downloaded {len(nodes)} nodes and {len(edges)} edges")

            # Save to files
            edges_path = self.processed_dir / "jakarta_roads.geojson"
            nodes_path = self.processed_dir / "jakarta_road_nodes.geojson"

            edges.to_file(edges_path, driver='GeoJSON')
            nodes.to_file(nodes_path, driver='GeoJSON')

            logger.success(f"Saved road network to: {edges_path}")

            return edges

        except Exception as e:
            logger.error(f"Road network extraction failed: {e}")
            raise

    def collect_all(self, use_osmnx: bool = True) -> dict:
        """
        Run complete OSM data collection pipeline

        Args:
            use_osmnx: If True, use OSMnx (simpler, no osmium needed)
                      If False, download full Indonesia extract (requires osmium)

        Returns:
            Dictionary with paths to collected data
        """
        logger.info("Starting OSM data collection for Jakarta...")

        results = {}

        if use_osmnx:
            # Simpler approach using OSMnx
            logger.info("Using OSMnx method (no osmium required)")
            pois = self.extract_pois_with_osmnx()
            results['pois'] = self.processed_dir / "jakarta_pois_osm.geojson"

            roads = self.extract_road_network()
            results['roads'] = self.processed_dir / "jakarta_roads.geojson"
            results['road_nodes'] = self.processed_dir / "jakarta_road_nodes.geojson"

        else:
            # Full download method (requires osmium-tool)
            logger.info("Using full download method (requires osmium-tool)")
            indonesia_pbf = self.download_indonesia_extract()
            results['indonesia_pbf'] = indonesia_pbf

            jakarta_pbf = self.extract_jakarta_pois_osmium(indonesia_pbf)
            results['jakarta_pbf'] = jakarta_pbf

            # Still use OSMnx for processing
            pois = self.extract_pois_with_osmnx()
            roads = self.extract_road_network()
            results['pois'] = self.processed_dir / "jakarta_pois_osm.geojson"
            results['roads'] = self.processed_dir / "jakarta_roads.geojson"

        logger.success("OSM data collection complete!")
        logger.info(f"Results: {results}")

        return results


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect OSM data for Jakarta')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--method', choices=['osmnx', 'full'], default='osmnx',
                       help='Collection method: osmnx (simple) or full (requires osmium)')

    args = parser.parse_args()

    collector = OSMCollector(data_dir=args.data_dir)
    results = collector.collect_all(use_osmnx=(args.method == 'osmnx'))

    print("\n" + "="*60)
    print("OSM Data Collection Complete!")
    print("="*60)
    for key, path in results.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
