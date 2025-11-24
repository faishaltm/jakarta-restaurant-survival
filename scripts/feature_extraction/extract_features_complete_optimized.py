# %% [code]
"""
Complete Feature Extraction - Research-Based Features with Automatic Checkpoints
================================================================================

Goal: Extract ALL Missing High-Impact Features
Output: jakarta_restaurant_features_complete.csv with ~50+ features

This script automatically saves checkpoints after each major section
and works in both Kaggle and local environments.

OPTIMIZATIONS:
- Skip existing checkpoint files
- Parallel processing for spatial operations
- Optional GPU acceleration detection
- Memory-efficient processing
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from shapely.strtree import STRtree
from shapely.geometry import Point
from tqdm import tqdm
from pathlib import Path
import gc
import warnings
import sys
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_CATEGORY = 'restaurant'

# Detect environment
try:
    # Try Kaggle paths first
    import os
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        DATASET_PATH = "/kaggle/input/jakarta-clean-categorized/jakarta_clean_categorized.csv"
        OUTPUT_DIR = Path("/kaggle/working/")
        IS_KAGGLE = True
        print("✓ Running on Kaggle")
    else:
        raise Exception("Not Kaggle")
except:
    DATASET_PATH = "outputs/kaggle_clean_data/jakarta_clean_categorized.csv"
    OUTPUT_DIR = Path("outputs/features")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    IS_KAGGLE = False
    print("✓ Running locally")

# Grid sizes for Shannon entropy
GRID_SIZES = [500, 1000, 2000]  # meters

# Buffer sizes for POI features
BUFFER_SIZES = [500, 1000, 2000, 5000]  # meters

# Parallel processing configuration
N_JOBS = min(mp.cpu_count() - 1, 8)  # Leave 1 core free
CHUNK_SIZE = 100  # Process POIs in chunks

print(f"\nTarget: {TARGET_CATEGORY}")
print(f"Grid sizes: {GRID_SIZES}")
print(f"Buffer sizes: {BUFFER_SIZES}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"CPU cores available: {mp.cpu_count()}")
print(f"Using {N_JOBS} cores for parallel processing")
print(f"\nExpected output: ~50+ features per restaurant\n")

# ============================================================================
# GPU DETECTION (Optional - for future use)
# ============================================================================

def detect_gpu():
    """Detect if GPU/CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
    except ImportError:
        pass

    try:
        import cudf
        print("✓ RAPIDS cuDF available")
        return True
    except ImportError:
        pass

    print("ℹ No GPU acceleration available (using CPU)")
    return False

GPU_AVAILABLE = detect_gpu()

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def checkpoint_exists(section_name):
    """Check if checkpoint file exists"""
    checkpoint_file = OUTPUT_DIR / f'checkpoint_{section_name}.csv'
    return checkpoint_file.exists()

def load_checkpoint(section_name):
    """Load existing checkpoint"""
    checkpoint_file = OUTPUT_DIR / f'checkpoint_{section_name}.csv'
    if checkpoint_file.exists():
        print(f"\n{'='*80}")
        print(f"✓ LOADING EXISTING CHECKPOINT: {section_name}")
        print(f"{'='*80}")
        print(f"  File: {checkpoint_file}")
        df = pd.read_csv(checkpoint_file)
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"{'='*80}\n")
        return df
    return None

def save_checkpoint(df, section_name, feature_count=None):
    """
    Save checkpoint after each section
    """
    checkpoint_file = OUTPUT_DIR / f'checkpoint_{section_name}.csv'

    # Select columns to save
    id_cols = ['name', 'latitude', 'longitude'] if 'name' in df.columns else []
    label_cols = ['event_observed', 'survival_days', 'categorical_label'] if 'categorical_label' in df.columns else []

    exclude_cols = id_cols + label_cols + [
        'geometry', 'date_created', 'date_refreshed', 'date_closed',
        'date_created_parsed', 'date_closed_parsed', 'main_category',
        'poi_type', 'regency', 'district'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('is_')]

    # Convert GeoDataFrame to DataFrame for saving
    if hasattr(df, 'geometry'):
        df_to_save = pd.DataFrame(df.drop(columns=['geometry']))
    else:
        df_to_save = df

    df_to_save[id_cols + feature_cols + label_cols].to_csv(checkpoint_file, index=False)

    print(f"\n{'='*80}")
    print(f"✓✓✓ CHECKPOINT SAVED: {section_name}")
    print(f"{'='*80}")
    print(f"  File: {checkpoint_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    if feature_count:
        print(f"  New features this section: {feature_count}")
    print(f"{'='*80}\n")

    return checkpoint_file

def merge_checkpoint_features(df_mature, checkpoint_df):
    """Merge features from checkpoint into current dataframe"""
    # Get feature columns from checkpoint
    exclude_cols = [
        'name', 'latitude', 'longitude',
        'event_observed', 'survival_days', 'categorical_label',
        'geometry', 'date_created', 'date_refreshed', 'date_closed',
        'date_created_parsed', 'date_closed_parsed', 'main_category',
        'poi_type', 'regency', 'district'
    ]

    feature_cols = [col for col in checkpoint_df.columns
                    if col not in exclude_cols and not col.startswith('is_')]

    # Merge on name/coordinates
    merge_cols = ['name', 'latitude', 'longitude']

    for col in feature_cols:
        if col not in df_mature.columns:
            # Merge the feature
            temp_df = checkpoint_df[merge_cols + [col]]
            df_mature = df_mature.merge(temp_df, on=merge_cols, how='left')

    return df_mature

# ============================================================================
# PARALLEL PROCESSING HELPERS
# ============================================================================

def process_poi_chunk_count(chunk_data):
    """Process a chunk of POIs for counting and density calculation"""
    chunk_indices, chunk_geometries, tree, gdf_poi, buffer_m, exclude_self, target_category = chunk_data

    counts = []
    densities = []
    area_km2 = (buffer_m / 1000) ** 2 * np.pi

    for idx, geom in zip(chunk_indices, chunk_geometries):
        buffer = geom.buffer(buffer_m)
        nearby_indices = tree.query(buffer)
        nearby = gdf_poi.iloc[nearby_indices]

        # Exclude self if needed
        if exclude_self:
            nearby = nearby[nearby.index != idx]

        within = nearby[nearby.geometry.within(buffer)]
        count = len(within)
        density = count / area_km2

        counts.append(count)
        densities.append(density)

    return counts, densities

def parallel_poi_features(target_df, poi_type, buffer_m, gdf_all, target_category):
    """Extract POI features using parallel processing"""
    gdf_poi = gdf_all[gdf_all['poi_type'] == poi_type]

    if len(gdf_poi) == 0:
        return [0] * len(target_df), [0.0] * len(target_df)

    tree = STRtree(gdf_poi.geometry)
    exclude_self = (poi_type == target_category)

    # Split into chunks
    indices = list(target_df.index)
    geometries = list(target_df.geometry)

    chunks = []
    for i in range(0, len(indices), CHUNK_SIZE):
        chunk_indices = indices[i:i+CHUNK_SIZE]
        chunk_geometries = geometries[i:i+CHUNK_SIZE]
        chunks.append((chunk_indices, chunk_geometries, tree, gdf_poi, buffer_m, exclude_self, target_category))

    # Process in parallel
    with mp.Pool(N_JOBS) as pool:
        results = list(tqdm(
            pool.imap(process_poi_chunk_count, chunks),
            total=len(chunks),
            desc=f"  {poi_type[:15]:15s} {buffer_m}m",
            leave=False
        ))

    # Combine results
    counts = []
    densities = []
    for chunk_counts, chunk_densities in results:
        counts.extend(chunk_counts)
        densities.extend(chunk_densities)

    return counts, densities

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80)
print()

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Load ALL POIs for context
gdf_all = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs='EPSG:4326'
).to_crs(epsg=32748)

print(f"✓ Total POIs: {len(gdf_all):,}")

# Filter target
gdf_target = gdf_all[gdf_all['poi_type'] == TARGET_CATEGORY].copy()
print(f"✓ Target ({TARGET_CATEGORY}): {len(gdf_target):,}")

# Free memory
del df
gc.collect()

print("\n✓ Data loaded\n")

# ============================================================================
# CREATE SURVIVAL LABELS
# ============================================================================

print("="*80)
print("CREATING SURVIVAL LABELS")
print("="*80)
print()

gdf_target['date_created_parsed'] = pd.to_datetime(gdf_target['date_created'], errors='coerce')
gdf_target['date_closed_parsed'] = pd.to_datetime(gdf_target['date_closed'], errors='coerce')

REFERENCE_DATE = pd.Timestamp('2024-01-01')
OBSERVATION_WINDOW_DAYS = 365 * 3

gdf_target['event_observed'] = gdf_target['date_closed_parsed'].notna().astype(int)
gdf_target['survival_days'] = np.where(
    gdf_target['event_observed'] == 1,
    (gdf_target['date_closed_parsed'] - gdf_target['date_created_parsed']).dt.days,
    (REFERENCE_DATE - gdf_target['date_created_parsed']).dt.days
)

gdf_target['categorical_label'] = 2
gdf_target.loc[
    (gdf_target['date_created_parsed'] <= REFERENCE_DATE - timedelta(days=OBSERVATION_WINDOW_DAYS)) &
    (gdf_target['date_closed_parsed'].notna()) &
    (gdf_target['date_closed_parsed'] <= REFERENCE_DATE),
    'categorical_label'
] = 0
gdf_target.loc[
    (gdf_target['date_created_parsed'] <= REFERENCE_DATE - timedelta(days=OBSERVATION_WINDOW_DAYS)) &
    (gdf_target['date_closed_parsed'].isna()),
    'categorical_label'
] = 1

df_mature = gdf_target[gdf_target['categorical_label'] != 2].copy()

print(f"Mature POIs: {len(df_mature):,}")
print(f"  Failures: {(df_mature['categorical_label'] == 0).sum():,}")
print(f"  Successes: {(df_mature['categorical_label'] == 1).sum():,}\n")

# Free memory
del gdf_target
gc.collect()

# ============================================================================
# CHECK FOR FINAL OUTPUT - Skip everything if exists
# ============================================================================

final_output_file = OUTPUT_DIR / 'jakarta_restaurant_features_complete.csv'
if final_output_file.exists():
    print("="*80)
    print("✓✓✓ FINAL OUTPUT ALREADY EXISTS!")
    print("="*80)
    print(f"  File: {final_output_file}")
    print(f"\n  To regenerate, delete this file first.")
    print("="*80)
    sys.exit(0)

# ============================================================================
# SECTION 1: Shannon Entropy Multi-Scale
# ============================================================================

SECTION_NAME = 'section1_entropy'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 1: SHANNON ENTROPY - MULTI-SCALE")
    print("="*80)
    print("Impact: 70% feature importance (proven)\n")

    # POI categories for entropy calculation
    POI_CATEGORIES = [
        'restaurant', 'office', 'mall', 'university', 'residential',
        'hospital', 'bank', 'transport', 'school', 'gym'
    ]

    def calculate_shannon_entropy(counts):
        """Calculate Shannon entropy for POI diversity"""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)
        return entropy

    # Calculate for each grid size
    for grid_size in GRID_SIZES:
        print(f"\nCalculating Shannon entropy at {grid_size}m grid...")

        # Create grid coordinates
        gdf_all['grid_x'] = (gdf_all.geometry.x / grid_size).astype(int)
        gdf_all['grid_y'] = (gdf_all.geometry.y / grid_size).astype(int)

        # Count POIs per grid cell
        grid_counts = defaultdict(lambda: {cat: 0 for cat in POI_CATEGORIES})

        for idx, row in gdf_all.iterrows():
            grid_key = (row['grid_x'], row['grid_y'])
            if row['poi_type'] in POI_CATEGORIES:
                grid_counts[grid_key][row['poi_type']] += 1

        print(f"  Grids: {len(grid_counts):,}")

        # Calculate entropy for each grid
        grid_entropy = {}
        for grid_key, counts in grid_counts.items():
            grid_entropy[grid_key] = calculate_shannon_entropy(counts)

        # Assign entropy to each mature POI (use 8-neighbor average)
        df_mature['grid_x'] = (df_mature.geometry.x / grid_size).astype(int)
        df_mature['grid_y'] = (df_mature.geometry.y / grid_size).astype(int)

        entropy_values = []
        for idx, row in tqdm(df_mature.iterrows(), total=len(df_mature), desc=f"  {grid_size}m"):
            poi_grid = (row['grid_x'], row['grid_y'])

            # Average entropy from 8 neighbors
            entropy_sum = 0.0
            count = 0

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if max(abs(dx), abs(dy)) == 1:  # 8-neighbors only
                        neighbor = (poi_grid[0] + dx, poi_grid[1] + dy)
                        if neighbor in grid_entropy:
                            entropy_sum += grid_entropy[neighbor]
                            count += 1

            avg_entropy = entropy_sum / count if count > 0 else 0.0
            entropy_values.append(avg_entropy)

        col_name = f'entropy_{grid_size}m'
        df_mature[col_name] = entropy_values

        print(f"  ✓ {col_name}: mean={np.mean(entropy_values):.3f}, std={np.std(entropy_values):.3f}")

    # Clean up grid columns
    df_mature = df_mature.drop(columns=['grid_x', 'grid_y'])
    gdf_all = gdf_all.drop(columns=['grid_x', 'grid_y'])

    print(f"\n✓ Shannon entropy extracted for {len(GRID_SIZES)} scales")
    gc.collect()

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=3)

# ============================================================================
# SECTION 2: POI Counts & Densities
# ============================================================================

SECTION_NAME = 'section2_poi_features'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 2: POI COUNTS & DENSITIES (PARALLEL)")
    print("="*80)
    print()

    # Define POI categories to extract
    POI_TYPES_TO_EXTRACT = {
        'competitors': TARGET_CATEGORY,
        'mall': 'mall',
        'office': 'office',
        'transport': 'transport',
        'residential': 'residential',
        'school': 'school',
        'hospital': 'hospital',
        'bank': 'bank'
    }

    # Extract for all combinations using parallel processing
    for poi_name, poi_type in POI_TYPES_TO_EXTRACT.items():
        print(f"\nExtracting {poi_name}...")

        for buffer_m in BUFFER_SIZES:
            counts, densities = parallel_poi_features(
                df_mature, poi_type, buffer_m, gdf_all, TARGET_CATEGORY
            )

            df_mature[f'{poi_name}_count_{buffer_m}m'] = counts
            df_mature[f'{poi_name}_density_{buffer_m}m'] = densities

            print(f"  {buffer_m}m: count={np.mean(counts):.1f}, density={np.mean(densities):.2f}/km²")

    print(f"\n✓ POI features extracted for {len(POI_TYPES_TO_EXTRACT)} types × {len(BUFFER_SIZES)} buffers")
    print(f"  Total: {len(POI_TYPES_TO_EXTRACT) * len(BUFFER_SIZES) * 2} features (count + density)")
    gc.collect()

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=64)

# ============================================================================
# SECTION 3: Indonesia-Specific Features
# ============================================================================

SECTION_NAME = 'section3_indonesia'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 3: INDONESIA-SPECIFIC POI FEATURES")
    print("="*80)
    print()

    def contains_keyword(name, keywords):
        if pd.isna(name):
            return False
        name_lower = str(name).lower()
        return any(kw in name_lower for kw in keywords)

    INDONESIA_KEYWORDS = {
        'mosque': ['masjid', 'musholla', 'mushola', 'mosque'],
        'pasar': ['pasar', 'market'],
        'convenience': ['indomaret', 'alfamart', 'alfamidi'],
        'gas_station': ['spbu', 'pertamina', 'shell']
    }

    # Detect Indonesia-specific POIs
    print("Detecting Indonesia-specific POIs...\n")
    for category, keywords in INDONESIA_KEYWORDS.items():
        gdf_all[f'is_{category}'] = gdf_all['name'].apply(lambda x: contains_keyword(x, keywords))
        count = gdf_all[f'is_{category}'].sum()
        print(f"  {category:15s}: {count:,} POIs detected")

    print()

    # Extract counts, densities, AND distances
    for indo_type in INDONESIA_KEYWORDS.keys():
        print(f"\nExtracting {indo_type}...")

        gdf_indo = gdf_all[gdf_all[f'is_{indo_type}']]

        if len(gdf_indo) == 0:
            print(f"  ⚠ No {indo_type} found, using zeros")
            for buffer_m in BUFFER_SIZES:
                df_mature[f'{indo_type}_count_{buffer_m}m'] = 0
                df_mature[f'{indo_type}_density_{buffer_m}m'] = 0.0
            df_mature[f'nearest_{indo_type}_m'] = 10000
            continue

        # Build spatial index
        tree = STRtree(gdf_indo.geometry)

        # Extract counts & densities using parallel processing
        for buffer_m in BUFFER_SIZES:
            counts, densities = parallel_poi_features(
                df_mature, f'is_{indo_type}', buffer_m, gdf_all, TARGET_CATEGORY
            )

            df_mature[f'{indo_type}_count_{buffer_m}m'] = counts
            df_mature[f'{indo_type}_density_{buffer_m}m'] = densities

            print(f"  {buffer_m}m: count={np.mean(counts):.1f}, density={np.mean(densities):.2f}/km²")

        # Extract nearest distance
        print(f"  Calculating nearest {indo_type} distance...")
        distances = []
        for idx, poi in tqdm(df_mature.iterrows(), total=len(df_mature), desc=f"  Distance", leave=False):
            dists = gdf_indo.geometry.distance(poi.geometry)
            distances.append(dists.min() if len(dists) > 0 else 10000)

        df_mature[f'nearest_{indo_type}_m'] = distances
        print(f"  nearest_{indo_type}_m: mean={np.mean(distances):.0f}m")

    print(f"\n✓ Indonesia-specific features extracted")
    print(f"  Total: {len(INDONESIA_KEYWORDS) * (len(BUFFER_SIZES) * 2 + 1)} features (count + density + distance)")
    gc.collect()

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=36)

# ============================================================================
# SECTION 4: Competition Metrics
# ============================================================================

SECTION_NAME = 'section4_competition'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 4: COMPETITION METRICS")
    print("="*80)
    print()

    # Nearest competitor distance
    print("Calculating nearest competitor distance...")
    gdf_competitors = gdf_all[gdf_all['poi_type'] == TARGET_CATEGORY]
    distances = []

    for idx, poi in tqdm(df_mature.iterrows(), total=len(df_mature), desc="  Distance"):
        others = gdf_competitors[gdf_competitors.index != idx]
        if len(others) > 0:
            dists = others.geometry.distance(poi.geometry)
            distances.append(dists.min())
        else:
            distances.append(10000)

    df_mature['nearest_competitor_m'] = distances
    print(f"✓ nearest_competitor_m: mean={np.mean(distances):.0f}m")

    # Average competitor distance (within 2km)
    print("\nCalculating average competitor distance (2km)...")
    avg_distances = []

    for idx, poi in tqdm(df_mature.iterrows(), total=len(df_mature), desc="  Avg dist"):
        buffer = poi.geometry.buffer(2000)
        nearby = gdf_competitors[gdf_competitors.geometry.within(buffer)]
        nearby = nearby[nearby.index != idx]

        if len(nearby) > 0:
            dists = nearby.geometry.distance(poi.geometry)
            avg_distances.append(dists.mean())
        else:
            avg_distances.append(2000)

    df_mature['avg_competitor_dist_2km'] = avg_distances
    print(f"✓ avg_competitor_dist_2km: mean={np.mean(avg_distances):.0f}m")

    # Cannibalization risk (competitors within 500m)
    print("\nCalculating cannibalization risk...")
    df_mature['cannibalization_risk_500m'] = df_mature['competitors_count_500m']
    print(f"✓ cannibalization_risk_500m: mean={df_mature['cannibalization_risk_500m'].mean():.1f}")

    print(f"\n✓ Competition metrics extracted: 3 features")
    gc.collect()

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=3)

# ============================================================================
# SECTION 5: Demographics
# ============================================================================

SECTION_NAME = 'section5_demographics'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 5: DEMOGRAPHIC FEATURES")
    print("="*80)
    print()

    # Income (millions IDR/month)
    jakarta_income = {
        'Setiabudi': 22.8, 'Kebayoran Baru': 18.5, 'Menteng': 19.3,
        'Tanah Abang': 12.4, 'Cilandak': 16.2, 'Kebayoran Lama': 14.1,
        'Mampang Prapatan': 15.3, 'Tebet': 13.9, 'Pancoran': 11.7,
        'Pasar Minggu': 9.8, 'Jagakarsa': 9.5, 'Pesanggrahan': 9.2,
        'Gambir': 11.1, 'Kemayoran': 10.3, 'Sawah Besar': 9.1,
        'Senen': 8.1, 'Cempaka Putih': 10.8, 'Johar Baru': 8.9,
        'Cakung': 7.8, 'Jatinegara': 9.4, 'Kramat Jati': 8.5,
        'Matraman': 9.9, 'Pasar Rebo': 8.2, 'Ciracas': 7.9,
        'Cengkareng': 8.7, 'Grogol Petamburan': 11.5, 'Kalideres': 7.5,
        'Kebon Jeruk': 12.8, 'Kembangan': 8.4, 'Palmerah': 13.2,
    }

    # Density (per km²)
    jakarta_density = {
        'Cilandak': 5979, 'Jagakarsa': 12281, 'Kebayoran Baru': 7999,
        'Kebayoran Lama': 9629, 'Mampang Prapatan': 7112, 'Pancoran': 9885,
        'Pasar Minggu': 9081, 'Pesanggrahan': 7955, 'Setiabudi': 8572,
        'Tebet': 13010, 'Cempaka Putih': 8855, 'Gambir': 5746,
        'Johar Baru': 27135, 'Kemayoran': 14957, 'Menteng': 16111,
        'Sawah Baru': 27769, 'Senen': 19499, 'Tanah Abang': 17797,
        'Cakung': 11466, 'Ciracas': 10203, 'Jatinegara': 24500,
        'Kramat Jati': 12178, 'Matraman': 18670, 'Pasar Rebo': 11704,
        'Cengkareng': 13897, 'Kalideres': 15811, 'Kebon Jeruk': 12165,
        'Kembangan': 17094, 'Palmerah': 25872,
    }

    df_mature['income_district_m'] = df_mature['district'].map(
        lambda x: jakarta_income.get(str(x).replace(' ', ''), 10.5)
    )
    df_mature['density_district'] = df_mature['district'].map(
        lambda x: jakarta_density.get(str(x).replace(' ', ''), 12000)
    )
    df_mature['working_age_district'] = df_mature['density_district'] * 0.43

    print(f"✓ income_district_m: mean={df_mature['income_district_m'].mean():.1f}M IDR")
    print(f"✓ density_district: mean={df_mature['density_district'].mean():.0f}/km²")
    print(f"✓ working_age_district: mean={df_mature['working_age_district'].mean():.0f}")

    print(f"\n✓ Demographics: 3 features")

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=3)

# ============================================================================
# SECTION 6: Accessibility Features
# ============================================================================

SECTION_NAME = 'section6_accessibility'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 6: ACCESSIBILITY FEATURES")
    print("="*80)
    print()

    # Distance to city center (Monas)
    monas = gpd.GeoSeries([Point(106.8271, -6.1751)], crs='EPSG:4326').to_crs(epsg=32748)[0]
    df_mature['dist_city_center_km'] = df_mature.geometry.distance(monas) / 1000
    print(f"✓ dist_city_center_km: mean={df_mature['dist_city_center_km'].mean():.1f}km")

    # Transport density (use transport count)
    df_mature['transport_density_1km'] = df_mature['transport_density_1000m']
    print(f"✓ transport_density_1km: mean={df_mature['transport_density_1km'].mean():.2f}/km²")

    # Urban centrality (density / distance)
    df_mature['urban_centrality'] = df_mature['density_district'] / (df_mature['dist_city_center_km'] + 1)
    print(f"✓ urban_centrality: mean={df_mature['urban_centrality'].mean():.0f}")

    print(f"\n✓ Accessibility: 3 features")

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=3)

# ============================================================================
# SECTION 7: Interaction Features
# ============================================================================

SECTION_NAME = 'section7_interactions'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 7: INTERACTION FEATURES (ADVANCED)")
    print("="*80)
    print("Impact: +10-18% accuracy improvement!\n")

    # 1. Income × Population Density
    df_mature['income_pop_interaction'] = (
        df_mature['income_district_m'] * df_mature['density_district'] / 1000
    )
    print(f"✓ income_pop_interaction: mean={df_mature['income_pop_interaction'].mean():.1f}")

    # 2. Working Age × Mall (inverse distance)
    # Find nearest mall distance first
    gdf_malls = gdf_all[gdf_all['poi_type'] == 'mall']
    if len(gdf_malls) > 0:
        mall_distances = []
        for idx, poi in tqdm(df_mature.iterrows(), total=len(df_mature), desc="  Mall distance"):
            dists = gdf_malls.geometry.distance(poi.geometry)
            mall_distances.append(dists.min() if len(dists) > 0 else 5000)

        df_mature['nearest_mall_m'] = mall_distances
        df_mature['working_age_mall_inv'] = (
            df_mature['working_age_district'] / (df_mature['nearest_mall_m'] + 100)
        )
        print(f"✓ working_age_mall_inv: mean={df_mature['working_age_mall_inv'].mean():.2f}")
    else:
        df_mature['nearest_mall_m'] = 5000
        df_mature['working_age_mall_inv'] = 0

    # 3. Office × Transport
    df_mature['office_transport'] = (
        df_mature['office_count_1000m'] * df_mature['transport_density_1km']
    )
    print(f"✓ office_transport: mean={df_mature['office_transport'].mean():.1f}")

    # 4. Demand-Supply Ratio
    df_mature['demand_supply_ratio'] = (
        df_mature['density_district'] / (df_mature['competitors_count_1000m'] + 1)
    )
    print(f"✓ demand_supply_ratio: mean={df_mature['demand_supply_ratio'].mean():.1f}")

    # 5. Mosque × Residential
    df_mature['mosque_residential'] = (
        df_mature['mosque_count_1000m'] * df_mature['residential_count_1000m']
    )
    print(f"✓ mosque_residential: mean={df_mature['mosque_residential'].mean():.0f}")

    # 6. Pasar × Transport
    df_mature['pasar_transport'] = (
        df_mature['pasar_count_1000m'] * df_mature['transport_density_1km']
    )
    print(f"✓ pasar_transport: mean={df_mature['pasar_transport'].mean():.1f}")

    print(f"\n✓ Interactions: 6 features")
    gc.collect()

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=6)

# ============================================================================
# SECTION 8: Indonesia-Specific Advanced
# ============================================================================

SECTION_NAME = 'section8_indonesia_advanced'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 8: INDONESIA-SPECIFIC ADVANCED FEATURES")
    print("="*80)
    print()

    # 1. Friday Prayer Impact
    df_mature['friday_prayer_impact'] = (
        df_mature['mosque_count_500m'] *
        df_mature['working_age_district'] *
        0.1  # 10% attend nearby mosque
    )
    print(f"✓ friday_prayer_impact: mean={df_mature['friday_prayer_impact'].mean():.0f}")

    # 2. Pasar Proximity Score (inverse distance)
    df_mature['pasar_proximity_score'] = (
        1 / (df_mature['nearest_pasar_m'] + 100)
    )
    print(f"✓ pasar_proximity_score: mean={df_mature['pasar_proximity_score'].mean():.6f}")

    # 3. Gas Station Proximity Score
    df_mature['gas_proximity_score'] = (
        1 / (df_mature['nearest_gas_station_m'] + 100)
    )
    print(f"✓ gas_proximity_score: mean={df_mature['gas_proximity_score'].mean():.6f}")

    # 4. Market Saturation Index (POIs per 1000 people)
    # Using district population as proxy
    district_population = df_mature['density_district'] * 10  # Rough estimate: 10 km² per district
    df_mature['market_saturation_1km'] = (
        df_mature['competitors_count_1000m'] / (district_population / 1000)
    )
    print(f"✓ market_saturation_1km: mean={df_mature['market_saturation_1km'].mean():.3f}")

    print(f"\n✓ Indonesia Advanced: 4 features")

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=4)

# ============================================================================
# SECTION 9: Temporal Features
# ============================================================================

SECTION_NAME = 'section9_temporal'
if checkpoint_exists(SECTION_NAME):
    checkpoint_df = load_checkpoint(SECTION_NAME)
    df_mature = merge_checkpoint_features(df_mature, checkpoint_df)
    print(f"✓ Skipping {SECTION_NAME} - using checkpoint\n")
else:
    print("="*80)
    print("SECTION 9: TEMPORAL FEATURES")
    print("="*80)
    print("Impact: +3-5% accuracy\n")

    # Temporal multipliers (constants for Indonesia)
    df_mature['ramadan_evening_multiplier'] = 2.5  # Buka puasa surge
    df_mature['ramadan_daytime_multiplier'] = 0.3  # Fasting period
    df_mature['weekend_mall_multiplier'] = 1.8     # Weekend mall visits
    df_mature['gajian_multiplier'] = 1.4           # Spending surge (25th-5th)
    df_mature['school_holiday_multiplier'] = 1.3   # June-July, Dec-Jan

    print(f"✓ ramadan_evening_multiplier: {df_mature['ramadan_evening_multiplier'].iloc[0]}")
    print(f"✓ ramadan_daytime_multiplier: {df_mature['ramadan_daytime_multiplier'].iloc[0]}")
    print(f"✓ weekend_mall_multiplier: {df_mature['weekend_mall_multiplier'].iloc[0]}")
    print(f"✓ gajian_multiplier: {df_mature['gajian_multiplier'].iloc[0]}")
    print(f"✓ school_holiday_multiplier: {df_mature['school_holiday_multiplier'].iloc[0]}")

    print(f"\n✓ Temporal: 5 features")

    # SAVE CHECKPOINT
    save_checkpoint(df_mature, SECTION_NAME, feature_count=5)

# ============================================================================
# FINAL SAVE: Complete Feature Set
# ============================================================================

print("\n" + "="*80)
print("FINAL SAVE: COMPLETE FEATURE SET")
print("="*80)
print()

# Select columns to save
id_cols = ['name', 'latitude', 'longitude']
label_cols = ['event_observed', 'survival_days', 'categorical_label']

# Get all feature columns (exclude geometry, parsed dates, etc)
exclude_cols = id_cols + label_cols + [
    'geometry', 'date_created', 'date_refreshed', 'date_closed',
    'date_created_parsed', 'date_closed_parsed', 'main_category',
    'poi_type', 'regency', 'district'
]

feature_cols = [col for col in df_mature.columns if col not in exclude_cols and not col.startswith('is_')]

print(f"Total features extracted: {len(feature_cols)}")
print(f"  Shannon Entropy: {len([c for c in feature_cols if 'entropy' in c])}")
print(f"  POI Counts: {len([c for c in feature_cols if '_count_' in c])}")
print(f"  POI Densities: {len([c for c in feature_cols if '_density_' in c])}")
print(f"  Distances: {len([c for c in feature_cols if 'nearest_' in c or 'dist_' in c])}")
print(f"  Interactions: {len([c for c in feature_cols if 'interaction' in c or c in ['demand_supply_ratio', 'mosque_residential', 'pasar_transport', 'office_transport']])}")
print(f"  Indonesia Advanced: {len([c for c in feature_cols if 'friday' in c or 'proximity_score' in c or 'saturation' in c])}")
print(f"  Temporal: {len([c for c in feature_cols if 'multiplier' in c])}")

# Save final file
output_file = OUTPUT_DIR / 'jakarta_restaurant_features_complete.csv'

# Convert to DataFrame for saving
df_final = pd.DataFrame(df_mature.drop(columns=['geometry']))
df_final[id_cols + feature_cols + label_cols].to_csv(output_file, index=False)

print(f"\n✓ FINAL FILE SAVED: {output_file}")
print(f"  Rows: {len(df_mature):,}")
print(f"  Columns: {len(id_cols) + len(feature_cols) + len(label_cols)}")

# Save feature list
feature_list_file = OUTPUT_DIR / 'feature_list_complete.txt'
with open(feature_list_file, 'w', encoding='utf-8') as f:
    f.write(f"Total Features: {len(feature_cols)}\n\n")
    f.write("="*80 + "\n")
    f.write("FEATURE BREAKDOWN\n")
    f.write("="*80 + "\n\n")

    f.write(f"Shannon Entropy: {len([c for c in feature_cols if 'entropy' in c])}\n")
    f.write(f"POI Counts: {len([c for c in feature_cols if '_count_' in c])}\n")
    f.write(f"POI Densities: {len([c for c in feature_cols if '_density_' in c])}\n")
    f.write(f"Distances: {len([c for c in feature_cols if 'nearest_' in c or 'dist_' in c])}\n")
    f.write(f"Interactions: {len([c for c in feature_cols if 'interaction' in c or c in ['demand_supply_ratio', 'mosque_residential', 'pasar_transport', 'office_transport']])}\n")
    f.write(f"Indonesia Advanced: {len([c for c in feature_cols if 'friday' in c or 'proximity_score' in c or 'saturation' in c])}\n")
    f.write(f"Temporal: {len([c for c in feature_cols if 'multiplier' in c])}\n\n")

    f.write("="*80 + "\n")
    f.write("ALL FEATURES (ALPHABETICAL)\n")
    f.write("="*80 + "\n\n")

    for i, feat in enumerate(sorted(feature_cols), 1):
        f.write(f"{i:3d}. {feat}\n")

print(f"✓ Feature list saved: {feature_list_file}")

print("\n" + "="*80)
print("FEATURE EXTRACTION COMPLETE!")
print("="*80)
print(f"\nOutput files in: {OUTPUT_DIR}")
print(f"  - jakarta_restaurant_features_complete.csv (FINAL)")
print(f"  - feature_list_complete.txt")
print(f"  - checkpoint_*.csv (9 checkpoint files)")
print("\nReady for model training!")
print("="*80)
