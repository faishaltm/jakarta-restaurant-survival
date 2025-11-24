"""
Create comparison data with contour lines/boundaries to show:
1. Success-only zones (blue boundary)
2. Failure-only zones (red boundary)
3. Overlapping zones (purple boundary)

Strategy: Create density grids and use contour levels to show boundaries
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import json
import numpy as np
from collections import defaultdict

print("Loading restaurant data...")
df = pd.read_csv('outputs/archive/jakarta_restaurant_phase1_2_5_combined.csv')

failures = df[df['date_closed'].notna()].copy()
successes = df[df['date_closed'].isna()].copy()

print(f"Failures: {len(failures):,}")
print(f"Successes: {len(successes):,}")

# Create grid-based density map
def create_density_grid(points_df, grid_size=0.005):
    """Create density grid with counts per cell"""
    grid = defaultdict(lambda: {'lat': 0, 'lon': 0, 'count': 0, 'points': []})

    for _, row in points_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        key = (round(lat / grid_size), round(lon / grid_size))
        grid[key]['lat'] += lat
        grid[key]['lon'] += lon
        grid[key]['count'] += 1
        grid[key]['points'].append([lat, lon])

    # Calculate average position for each cell
    result = {}
    for key, cell in grid.items():
        avg_lat = cell['lat'] / cell['count']
        avg_lon = cell['lon'] / cell['count']
        result[key] = {
            'center': [round(avg_lat, 6), round(avg_lon, 6)],
            'count': cell['count'],
            'intensity': min(cell['count'] / 10, 1.0)
        }

    return result

print("\n📊 Creating density grids...")

# Create grids at different resolutions
grid_sizes = {
    'low': 0.01,      # ~1.1km
    'medium': 0.005,  # ~550m
    'high': 0.003     # ~330m
}

output_data = {
    'zoom_data': {},
    'stats': {
        'total_failures': len(failures),
        'total_successes': len(successes)
    }
}

for zoom_level, grid_size in grid_sizes.items():
    print(f"\nProcessing {zoom_level} zoom (grid size: {grid_size})...")

    # Create grids for both datasets
    success_grid = create_density_grid(successes, grid_size)
    failure_grid = create_density_grid(failures, grid_size)

    # Find all grid cells
    all_keys = set(success_grid.keys()) | set(failure_grid.keys())

    # Categorize cells
    success_only = []
    failure_only = []
    overlap = []

    for key in all_keys:
        has_success = key in success_grid
        has_failure = key in failure_grid

        if has_success and has_failure:
            # Overlapping zone - combine intensity
            center = success_grid[key]['center']
            s_intensity = success_grid[key]['intensity']
            f_intensity = failure_grid[key]['intensity']
            # Higher intensity = more data
            combined_intensity = min((s_intensity + f_intensity) / 2, 1.0)
            overlap.append([center[0], center[1], combined_intensity])
        elif has_success:
            # Success-only zone
            center = success_grid[key]['center']
            intensity = success_grid[key]['intensity']
            success_only.append([center[0], center[1], intensity])
        else:
            # Failure-only zone
            center = failure_grid[key]['center']
            intensity = failure_grid[key]['intensity']
            failure_only.append([center[0], center[1], intensity])

    output_data['zoom_data'][zoom_level] = {
        'success_only': success_only,
        'failure_only': failure_only,
        'overlap': overlap
    }

    print(f"  Success-only cells: {len(success_only)}")
    print(f"  Failure-only cells: {len(failure_only)}")
    print(f"  Overlapping cells: {len(overlap)}")

# Save to JS file
output_path = 'outputs/visualizations/restaurant_contour_data.js'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('const restaurantContourData = ')
    json.dump(output_data, f, separators=(',', ':'))
    f.write(';')

file_size = len(json.dumps(output_data)) / 1024

print(f"\n✅ Success! Created {output_path}")
print(f"   File size: {file_size:.1f} KB")
print(f"\n📈 Data structure:")
print(f"   - 3 zoom levels (low, medium, high)")
print(f"   - 3 categories per level (success_only, failure_only, overlap)")
print(f"   - Each point: [latitude, longitude, intensity]")
