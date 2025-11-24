"""
POI Project Directory Cleanup and Reorganization
=================================================

This script:
1. Removes obsolete/unused files
2. Organizes files into clean directory structure
3. Keeps only essential files for the project
"""

import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

print("="*80)
print("POI PROJECT CLEANUP & REORGANIZATION")
print("="*80)
print()

# ============================================================================
# STEP 1: Define what to keep and what to remove
# ============================================================================

# Files to KEEP in root
KEEP_ROOT_FILES = {
    # Main production script
    'kaggle_feature_extraction_with_checkpoints.py',

    # Requirements
    'requirements.txt',

    # Notebooks (active/useful)
    'kaggle_feature_extraction_complete.ipynb',
    'kaggle_survival_training_advanced.ipynb',
    'kaggle_feature_importance_analysis.ipynb',
}

# Files to ARCHIVE (move to archive folder)
ARCHIVE_ROOT_FILES = {
    # Old phase notebooks
    'kaggle_phase1_demographics.ipynb',
    'kaggle_phase2_competition.ipynb',
    'kaggle_phase3_accessibility.ipynb',
    'kaggle_phase4_indonesia_specific.ipynb',
    'kaggle_phases_all_in_one.ipynb',
    'phase5_interactions_local.ipynb',

    # Old experimental notebooks
    'kaggle_simple_2features.ipynb',
    'kaggle_single_category_multilayer.ipynb',
    'kaggle_survival_prediction_thematic.ipynb',
    'kaggle_all_features_importance.ipynb',
    'kaggle_high_impact_features.ipynb',
}

# Old data collection scripts (can be removed or archived)
OLD_COLLECTION_SCRIPTS = {
    'add_demographic_features_kaggle.py',
    'add_regions_to_clean_dataset.py',
    'assign_nearest_boundaries.py',
    'collect_boundaries.py',
    'collect_buildings.py',
    'collect_foursquare_os.py',
    'collect_osm_buildings.py',
    'collect_population.py',
    'create_clean_categorized_dataset.py',
    'create_clean_raw_dataset.py',
    'create_clean_raw_dataset_with_regions.py',
    'create_grid_features.py',
    'test_osm_roads.py',
    'test_survival_labeling_all_pois.py',
    'setup_check.py',
    'show_poi_examples.py',
    'verify_data_structure.py',
    'visualize_district_verification.py',
}

# Output directories to clean/organize
OUTPUT_CLEAN_STRUCTURE = {
    'outputs/final_dataset': 'Final production-ready datasets',
    'outputs/archive': 'Old/experimental outputs',
    'archive/notebooks': 'Old notebooks',
    'archive/scripts': 'Old data collection scripts',
}

# ============================================================================
# STEP 2: Create new directory structure
# ============================================================================

print("Creating new directory structure...")
for dir_path, description in OUTPUT_CLEAN_STRUCTURE.items():
    full_path = BASE_DIR / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {dir_path} ({description})")

print()

# ============================================================================
# STEP 3: Archive old notebooks
# ============================================================================

print("Archiving old notebooks...")
archive_notebooks_dir = BASE_DIR / 'archive' / 'notebooks'
archived_count = 0

for notebook in ARCHIVE_ROOT_FILES:
    src = BASE_DIR / notebook
    if src.exists():
        dst = archive_notebooks_dir / notebook
        shutil.move(str(src), str(dst))
        print(f"  Archived: {notebook}")
        archived_count += 1

print(f"  Total archived: {archived_count} notebooks\n")

# ============================================================================
# STEP 4: Archive old scripts
# ============================================================================

print("Archiving old data collection scripts...")
archive_scripts_dir = BASE_DIR / 'archive' / 'scripts'
archived_scripts_count = 0

for script in OLD_COLLECTION_SCRIPTS:
    src = BASE_DIR / script
    if src.exists():
        dst = archive_scripts_dir / script
        shutil.move(str(src), str(dst))
        print(f"  Archived: {script}")
        archived_scripts_count += 1

print(f"  Total archived: {archived_scripts_count} scripts\n")

# ============================================================================
# STEP 5: Organize output files
# ============================================================================

print("Organizing output files...")

# Keep only the main clean dataset
outputs_dir = BASE_DIR / 'outputs'
final_dataset_dir = BASE_DIR / 'outputs' / 'final_dataset'
archive_outputs_dir = BASE_DIR / 'outputs' / 'archive'

# Move intermediate outputs to archive
intermediate_outputs = [
    'outputs/jakarta_restaurant_phase1_2_5_combined.csv',
    'outputs/jakarta_restaurant_phase1_demographics.csv',
    'outputs/features/coffee_shops_with_features.csv',
    'outputs/results/feature_importance.csv',
    'outputs/survival_analysis',
    'outputs/survival_analysis_jaksel',
    'outputs/survival_analysis_jaksel_fast',
]

for item in intermediate_outputs:
    src = BASE_DIR / item
    if src.exists():
        dst = archive_outputs_dir / Path(item).name
        if src.is_dir():
            shutil.move(str(src), str(dst))
            print(f"  Archived directory: {item}")
        else:
            shutil.move(str(src), str(dst))
            print(f"  Archived: {item}")

print()

# ============================================================================
# STEP 6: Clean up empty directories
# ============================================================================

print("Cleaning up empty directories...")

def remove_empty_dirs(path):
    """Recursively remove empty directories"""
    removed = []
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                removed.append(dirpath)
            except:
                pass
    return removed

removed_dirs = remove_empty_dirs(BASE_DIR / 'outputs')
for d in removed_dirs:
    print(f"  Removed empty: {d}")

if not removed_dirs:
    print("  No empty directories to remove")

print()

# ============================================================================
# STEP 7: Summary
# ============================================================================

print("="*80)
print("CLEANUP COMPLETE!")
print("="*80)
print()
print("Current directory structure:")
print()
print("POI/")
print("├── kaggle_feature_extraction_with_checkpoints.py  (MAIN SCRIPT)")
print("├── kaggle_feature_extraction_complete.ipynb       (Main notebook)")
print("├── kaggle_survival_training_advanced.ipynb        (Training notebook)")
print("├── kaggle_feature_importance_analysis.ipynb       (Analysis notebook)")
print("├── requirements.txt")
print("│")
print("├── src/                           (Source code modules)")
print("│   ├── data/")
print("│   ├── features/")
print("│   ├── models/")
print("│   └── utils/")
print("│")
print("├── data/processed/                (Raw input data)")
print("│   ├── foursquare/")
print("│   ├── osm/")
print("│   ├── buildings/")
print("│   └── bps/")
print("│")
print("├── outputs/")
print("│   ├── kaggle_clean_data/")
print("│   │   └── jakarta_clean_categorized.csv  (MAIN DATASET)")
print("│   ├── kaggle_raw_data/")
print("│   ├── final_dataset/             (For final outputs)")
print("│   └── archive/                   (Old outputs)")
print("│")
print("├── archive/")
print("│   ├── notebooks/                 (Old/experimental notebooks)")
print("│   └── scripts/                   (Old data collection scripts)")
print("│")
print("└── notebooks/                     (Original exploratory notebooks)")
print()

print("Files kept in root:")
for f in sorted(KEEP_ROOT_FILES):
    if (BASE_DIR / f).exists():
        print(f"  ✓ {f}")

print()
print(f"Files archived: {archived_count + archived_scripts_count}")
print()
print("="*80)
print()
print("Next steps:")
print("  1. Run kaggle_feature_extraction_with_checkpoints.py to generate features")
print("  2. Output will be saved to outputs/features/")
print("  3. Use kaggle_survival_training_advanced.ipynb for model training")
print()
print("="*80)
