"""
PostgreSQL + PostGIS Database Initialization

Creates database schema for Jakarta POI location intelligence platform.

Tables:
- pois: Point of interest locations
- coffee_shops: Training data (known coffee shop locations)
- administrative_boundaries: Jakarta kelurahan/kecamatan boundaries
- demographics: Demographic data by region
- features: Engineered features for each location
- predictions: Model predictions and scores
"""

import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'dbname': os.getenv('DB_NAME', 'jakarta_poi')
}


class DatabaseInitializer:
    """Initialize PostgreSQL database with PostGIS extension"""

    def __init__(self, config: dict = None):
        self.config = config or DB_CONFIG
        self.dbname = self.config['dbname']

    def create_database(self):
        """Create the database if it doesn't exist"""
        # Connect to default postgres database
        conn_config = self.config.copy()
        conn_config['dbname'] = 'postgres'

        try:
            logger.info(f"Connecting to PostgreSQL server at {conn_config['host']}...")

            conn = psycopg2.connect(**conn_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()

            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.dbname,)
            )

            exists = cur.fetchone()

            if exists:
                logger.info(f"Database '{self.dbname}' already exists")
            else:
                logger.info(f"Creating database '{self.dbname}'...")
                cur.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.dbname)
                ))
                logger.success(f"Database '{self.dbname}' created successfully")

            cur.close()
            conn.close()

        except psycopg2.Error as e:
            logger.error(f"Database creation failed: {e}")
            raise

    def create_extensions(self):
        """Create PostGIS and related extensions"""
        try:
            logger.info("Creating PostGIS extensions...")

            conn = psycopg2.connect(**self.config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()

            # Create PostGIS extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            logger.success("PostGIS extension created")

            # Create PostGIS Topology extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
            logger.success("PostGIS Topology extension created")

            # Create pg_trgm for text search
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            logger.success("pg_trgm extension created")

            # Check PostGIS version
            cur.execute("SELECT PostGIS_Version();")
            version = cur.fetchone()[0]
            logger.info(f"PostGIS version: {version}")

            cur.close()
            conn.close()

        except psycopg2.Error as e:
            logger.error(f"Extension creation failed: {e}")
            raise

    def create_schema(self):
        """Create all necessary tables"""
        logger.info("Creating database schema...")

        # SQL statements for creating tables
        schema_sql = """
        -- POIs table (all points of interest)
        CREATE TABLE IF NOT EXISTS pois (
            id SERIAL PRIMARY KEY,
            source VARCHAR(50),  -- 'osm', 'foursquare', 'google'
            external_id VARCHAR(255),
            name VARCHAR(255),
            category VARCHAR(100),
            subcategory VARCHAR(100),
            address TEXT,
            city VARCHAR(100),
            region VARCHAR(100),
            postal_code VARCHAR(20),
            rating DECIMAL(3, 2),
            rating_count INTEGER,
            geom GEOMETRY(Point, 4326),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create spatial index on geometry column
        CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_pois_category ON pois (category);
        CREATE INDEX IF NOT EXISTS idx_pois_source ON pois (source);

        -- Coffee shops table (training data)
        CREATE TABLE IF NOT EXISTS coffee_shops (
            id SERIAL PRIMARY KEY,
            brand VARCHAR(100),
            name VARCHAR(255),
            source VARCHAR(50),
            external_id VARCHAR(255),
            address TEXT,
            rating DECIMAL(3, 2),
            rating_count INTEGER,
            is_successful BOOLEAN DEFAULT TRUE,  -- Positive training example
            geom GEOMETRY(Point, 4326),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_coffee_shops_geom ON coffee_shops USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_coffee_shops_brand ON coffee_shops (brand);

        -- Administrative boundaries
        CREATE TABLE IF NOT EXISTS administrative_boundaries (
            id SERIAL PRIMARY KEY,
            level VARCHAR(50),  -- 'province', 'regency', 'district', 'village'
            code VARCHAR(50),
            name VARCHAR(255),
            parent_code VARCHAR(50),
            geom GEOMETRY(MultiPolygon, 4326),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_admin_geom ON administrative_boundaries USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_admin_level ON administrative_boundaries (level);
        CREATE INDEX IF NOT EXISTS idx_admin_code ON administrative_boundaries (code);

        -- Demographics table
        CREATE TABLE IF NOT EXISTS demographics (
            id SERIAL PRIMARY KEY,
            region_code VARCHAR(50),
            region_name VARCHAR(255),
            region_level VARCHAR(50),
            population INTEGER,
            population_density DECIMAL(10, 2),
            median_income DECIMAL(15, 2),
            median_age DECIMAL(5, 2),
            working_age_pct DECIMAL(5, 2),  -- Age 25-54 percentage
            education_high_pct DECIMAL(5, 2),
            employment_rate DECIMAL(5, 2),
            year INTEGER,
            geom GEOMETRY(MultiPolygon, 4326),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_demographics_geom ON demographics USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_demographics_code ON demographics (region_code);

        -- Features table (engineered features for ML)
        CREATE TABLE IF NOT EXISTS features (
            id SERIAL PRIMARY KEY,
            location_id INTEGER,  -- Reference to candidate location
            geom GEOMETRY(Point, 4326),

            -- Spatial features
            competitor_count_500m INTEGER,
            competitor_count_1km INTEGER,
            competitor_count_2km INTEGER,
            nearest_competitor_dist DECIMAL(10, 2),
            same_brand_count_2km INTEGER,
            poi_diversity_500m DECIMAL(5, 4),
            poi_diversity_1km DECIMAL(5, 4),

            -- Demographic features
            population_500m INTEGER,
            population_1km INTEGER,
            population_2km INTEGER,
            median_income_region DECIMAL(15, 2),
            working_age_pct DECIMAL(5, 2),

            -- Accessibility features
            distance_to_main_road DECIMAL(10, 2),
            distance_to_transit DECIMAL(10, 2),
            road_intersection_count_500m INTEGER,
            drive_time_10min_population INTEGER,
            walkability_score DECIMAL(5, 2),

            -- Synthetic mobility features
            estimated_daily_visitors INTEGER,
            morning_traffic_score DECIMAL(5, 2),
            afternoon_traffic_score DECIMAL(5, 2),
            evening_traffic_score DECIMAL(5, 2),

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_features_geom ON features USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_features_location ON features (location_id);

        -- Predictions table (model outputs)
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            geom GEOMETRY(Point, 4326),
            model_version VARCHAR(50),
            score DECIMAL(5, 2),  -- 0-100 suitability score
            probability DECIMAL(5, 4),  -- 0-1 probability
            percentile DECIMAL(5, 2),  -- Percentile ranking
            confidence_lower DECIMAL(5, 4),
            confidence_upper DECIMAL(5, 4),
            top_features JSONB,  -- Top contributing features
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_geom ON predictions USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_predictions_score ON predictions (score DESC);
        CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_version);

        -- Model performance tracking
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(50),
            model_type VARCHAR(50),
            training_date TIMESTAMP,
            n_training_samples INTEGER,
            accuracy DECIMAL(5, 4),
            precision_score DECIMAL(5, 4),
            recall DECIMAL(5, 4),
            auc_score DECIMAL(5, 4),
            f1_score DECIMAL(5, 4),
            cv_method VARCHAR(100),
            hyperparameters JSONB,
            feature_importance JSONB,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_model_perf_version ON model_performance (model_version);
        CREATE INDEX IF NOT EXISTS idx_model_perf_date ON model_performance (training_date DESC);
        """

        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()

            # Execute schema creation
            cur.execute(schema_sql)
            conn.commit()

            logger.success("Database schema created successfully")

            # List all tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)

            tables = cur.fetchall()
            logger.info("Created tables:")
            for table in tables:
                logger.info(f"  - {table[0]}")

            cur.close()
            conn.close()

        except psycopg2.Error as e:
            logger.error(f"Schema creation failed: {e}")
            raise

    def initialize(self):
        """Run complete database initialization"""
        logger.info("="*60)
        logger.info("PostgreSQL + PostGIS Database Initialization")
        logger.info("="*60)

        try:
            # Step 1: Create database
            self.create_database()

            # Step 2: Create extensions
            self.create_extensions()

            # Step 3: Create schema
            self.create_schema()

            logger.success("="*60)
            logger.success("Database initialization complete!")
            logger.success("="*60)

            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Initialize PostgreSQL database')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--user', default='postgres', help='Database user')
    parser.add_argument('--password', help='Database password')
    parser.add_argument('--dbname', default='jakarta_poi', help='Database name')

    args = parser.parse_args()

    # Build config from arguments
    config = {
        'host': args.host,
        'port': args.port,
        'user': args.user,
        'password': args.password or os.getenv('DB_PASSWORD', ''),
        'dbname': args.dbname
    }

    # Initialize database
    initializer = DatabaseInitializer(config)
    success = initializer.initialize()

    if success:
        print("\nDatabase is ready for use!")
        print(f"Connection string: postgresql://{config['user']}@{config['host']}:{config['port']}/{config['dbname']}")
    else:
        print("\nDatabase initialization failed. Check logs for details.")
        exit(1)


if __name__ == "__main__":
    main()
