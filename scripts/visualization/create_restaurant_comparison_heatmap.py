"""
Create optimized heatmap comparing restaurant failures vs successes.

Strategy for handling large success dataset (72,879 points):
1. Keep all 5,039 failures (important to see all failure patterns)
2. Use stratified spatial sampling for successes:
   - Low zoom: 500 representative points (grid-based sampling)
   - Medium zoom: 2,000 points
   - High zoom: 5,000 points (same order of magnitude as failures)
3. Grid-based aggregation to ensure geographic distribution
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import json
from collections import defaultdict

print("Loading restaurant data...")
df = pd.read_csv('outputs/archive/jakarta_restaurant_phase1_2_5_combined.csv')

print(f"Total restaurants: {len(df):,}")
print(f"Failures: {df['date_closed'].notna().sum():,}")
print(f"Successes: {df['date_closed'].isna().sum():,}")

# Split into failures and successes
failures = df[df['date_closed'].notna()].copy()
successes = df[df['date_closed'].isna()].copy()

print(f"\n✓ Loaded {len(failures):,} failures and {len(successes):,} successes")

def aggregate_with_intensity(points_df, grid_size):
    """Aggregate points using grid-based clustering with intensity."""
    grid = defaultdict(lambda: {'lat': 0, 'lon': 0, 'count': 0})

    for _, row in points_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        key = (round(lat / grid_size), round(lon / grid_size))
        grid[key]['lat'] += lat
        grid[key]['lon'] += lon
        grid[key]['count'] += 1

    result = []
    for cell in grid.values():
        avg_lat = round(cell['lat'] / cell['count'], 6)
        avg_lon = round(cell['lon'] / cell['count'], 6)
        intensity = min(cell['count'] / 5, 1.0)  # Normalize intensity
        result.append([avg_lat, avg_lon, intensity])

    return result

def stratified_sample(points_df, target_size, grid_size):
    """Sample points while maintaining geographic distribution."""
    # First aggregate to grid cells
    grid = defaultdict(list)

    for idx, row in points_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        key = (round(lat / grid_size), round(lon / grid_size))
        grid[key].append((lat, lon))

    # Sample from each grid cell proportionally
    total_cells = len(grid)
    points_per_cell = max(1, target_size // total_cells)

    sampled = []
    for cell_points in grid.values():
        # Take up to points_per_cell from each cell
        n_samples = min(len(cell_points), points_per_cell)
        sampled.extend(cell_points[:n_samples])

        if len(sampled) >= target_size:
            break

    # Format as [lat, lon, intensity]
    result = [[round(lat, 6), round(lon, 6), 0.5] for lat, lon in sampled[:target_size]]
    return result

print("\n📊 Processing FAILURES (all data - 5,039 points)...")

# For failures: use all data with intensity aggregation (same as before)
failures_data = {
    'low': aggregate_with_intensity(failures, grid_size=0.01),      # ~1.1km
    'medium': aggregate_with_intensity(failures, grid_size=0.003),  # ~333m
    'high': aggregate_with_intensity(failures, grid_size=0.001)     # ~111m
}

print(f"  Low zoom: {len(failures_data['low'])} points")
print(f"  Medium zoom: {len(failures_data['medium'])} points")
print(f"  High zoom: {len(failures_data['high'])} points")

print("\n📊 Processing SUCCESSES (sampled - from 72,879 points)...")

# For successes: use stratified sampling to keep dataset manageable
successes_data = {
    'low': stratified_sample(successes, target_size=500, grid_size=0.01),
    'medium': stratified_sample(successes, target_size=2000, grid_size=0.005),
    'high': stratified_sample(successes, target_size=5000, grid_size=0.002)
}

print(f"  Low zoom: {len(successes_data['low'])} points (from 72,879)")
print(f"  Medium zoom: {len(successes_data['medium'])} points")
print(f"  High zoom: {len(successes_data['high'])} points")

# Create output data structure
output_data = {
    'zoom_data': {
        'failures': failures_data,
        'successes': successes_data
    },
    'stats': {
        'total_failures': len(failures),
        'total_successes': len(successes),
        'failure_rate': round(len(failures) / len(df) * 100, 1),
        'success_rate': round(len(successes) / len(df) * 100, 1)
    }
}

# Save to JS file
output_path = 'outputs/visualizations/restaurant_comparison_data.js'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('const restaurantData = ')
    json.dump(output_data, f, separators=(',', ':'))
    f.write(';')

file_size = len(json.dumps(output_data)) / 1024

print(f"\n✅ Success! Created {output_path}")
print(f"   File size: {file_size:.1f} KB")
print(f"\n📈 Summary:")
print(f"   Failures: {len(failures):,} → {len(failures_data['high'])} aggregated points")
print(f"   Successes: {len(successes):,} → {len(successes_data['high'])} sampled points")
print(f"   Reduction: {(1 - (len(failures_data['high']) + len(successes_data['high'])) / len(df)) * 100:.1f}% smaller")
