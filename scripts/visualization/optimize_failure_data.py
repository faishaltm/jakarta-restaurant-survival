"""
Optimize Failure Data for Faster Loading
=========================================

Compress coordinates untuk reduce file size dan speed up loading.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("="*80)
print("OPTIMIZING FAILURE DATA")
print("="*80)
print()

# Load data
DATA_PATH = Path("outputs/kaggle_clean_data/jakarta_clean_categorized.csv")
df = pd.read_csv(DATA_PATH)

failures = df[df['date_closed'].notna()].copy()
print(f"Total failures: {len(failures):,}")

# Get top categories
category_stats = failures['poi_type'].value_counts().head(10)

# Optimize data format - round coordinates to 6 decimals (11cm precision)
# This reduces JSON size significantly

optimized_data = {}
category_info = {}

print("\nOptimizing coordinates...")
for category in category_stats.index:
    cat_failures = failures[failures['poi_type'] == category]

    # Round to 6 decimals for 11cm precision (enough for heatmap)
    coords = [
        [round(lat, 6), round(lon, 6)]
        for lat, lon in cat_failures[['latitude', 'longitude']].values
    ]

    optimized_data[category] = coords

    total_in_cat = len(df[df['poi_type'] == category])
    failure_count = len(cat_failures)
    failure_rate = failure_count / total_in_cat * 100 if total_in_cat > 0 else 0

    category_info[category] = {
        'count': failure_count,
        'total': total_in_cat,
        'rate': round(failure_rate, 2)
    }

    print(f"  ✓ {category}: {len(coords):,} points")

# Create compact JSON
export_data = {
    'data': optimized_data,
    'info': category_info,
    'stats': {
        'total': len(df),
        'failures': len(failures),
        'rate': round(len(failures) / len(df) * 100, 2)
    }
}

# Save compact version
output_path = Path("outputs/visualizations/failure_data_compact.js")

with open(output_path, 'w', encoding='utf-8') as f:
    # Use compact JSON (no whitespace)
    json_str = json.dumps(export_data, separators=(',', ':'))
    f.write(f"const failureData = {json_str};")

old_size = Path("outputs/visualizations/failure_data.js").stat().st_size
new_size = output_path.stat().st_size
reduction = (1 - new_size / old_size) * 100

print()
print("="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)
print(f"Original size: {old_size / 1024:.1f} KB")
print(f"Optimized size: {new_size / 1024:.1f} KB")
print(f"Reduction: {reduction:.1f}%")
print(f"\nOutput: {output_path}")
print("="*80)
