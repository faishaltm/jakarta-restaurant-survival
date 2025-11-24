"""
Aggressive Cleanup - Minimal Root Directory
============================================

Keep only absolute essentials in root.
Move everything else to organized subdirectories.
"""

import os
import shutil
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).parent

print("="*80)
print("AGGRESSIVE CLEANUP - MINIMAL ROOT DIRECTORY")
print("="*80)
print()

# ============================================================================
# STEP 1: Create organized structure
# ============================================================================

print("Creating organized directory structure...")

DIRS_TO_CREATE = {
    'docs': 'All documentation files',
    'notebooks': 'All Jupyter notebooks',
    'scripts': 'Utility scripts',
}

for dir_name, description in DIRS_TO_CREATE.items():
    dir_path = BASE_DIR / dir_name
    dir_path.mkdir(exist_ok=True)
    print(f"  ✓ {dir_name}/ ({description})")

print()

# ============================================================================
# STEP 2: Move all documentation to docs/
# ============================================================================

print("Moving documentation files to docs/...")

# All .md files except main README
md_files = list(BASE_DIR.glob('*.md'))
moved_docs = 0

for md_file in md_files:
    if md_file.name != 'README.md':  # Keep only main README in root
        dst = BASE_DIR / 'docs' / md_file.name
        shutil.move(str(md_file), str(dst))
        print(f"  → docs/{md_file.name}")
        moved_docs += 1

print(f"  Total: {moved_docs} files\n")

# ============================================================================
# STEP 3: Move all notebooks to notebooks/
# ============================================================================

print("Moving notebooks to notebooks/...")

# Move all .ipynb files from root
ipynb_files = list(BASE_DIR.glob('*.ipynb'))
moved_notebooks = 0

for notebook in ipynb_files:
    dst = BASE_DIR / 'notebooks' / notebook.name
    shutil.move(str(notebook), str(dst))
    print(f"  → notebooks/{notebook.name}")
    moved_notebooks += 1

print(f"  Total: {moved_notebooks} files\n")

# ============================================================================
# STEP 4: Move utility scripts to scripts/
# ============================================================================

print("Moving utility scripts to scripts/...")

UTILITY_SCRIPTS = [
    'cleanup_and_organize.py',
    'aggressive_cleanup.py',  # This script itself
]

moved_scripts = 0
for script in UTILITY_SCRIPTS:
    src = BASE_DIR / script
    if src.exists() and script != 'aggressive_cleanup.py':  # Don't move self yet
        dst = BASE_DIR / 'scripts' / script
        shutil.move(str(src), str(dst))
        print(f"  → scripts/{script}")
        moved_scripts += 1

print(f"  Total: {moved_scripts} files\n")

# ============================================================================
# STEP 5: Create a simple README
# ============================================================================

print("Creating clean README.md...")

readme_content = """# Jakarta POI Survival Analysis

Point of Interest survival prediction using spatial features and survival analysis.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Features (60-90 min)
```bash
python extract_features.py
```

### 3. Train Model
```bash
jupyter notebook notebooks/kaggle_survival_training_advanced.ipynb
```

---

## Project Structure

```
POI/
├── README.md                    This file
├── requirements.txt             Python dependencies
├── extract_features.py          Main feature extraction script
│
├── notebooks/                   Jupyter notebooks
│   ├── kaggle_feature_extraction_complete.ipynb
│   ├── kaggle_survival_training_advanced.ipynb
│   └── kaggle_feature_importance_analysis.ipynb
│
├── src/                        Source code modules
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── data/processed/             Raw input data
│
├── outputs/                    Generated outputs
│   ├── kaggle_clean_data/      Main dataset (27MB)
│   └── features/               Extracted features
│
├── docs/                       Documentation
│   └── PROJECT_STRUCTURE.md    Detailed structure
│
├── scripts/                    Utility scripts
└── archive/                    Archived files
```

---

## Dataset

**Main Dataset:** `outputs/kaggle_clean_data/jakarta_clean_categorized.csv`
- 158,377 POIs total
- 77,918 restaurants
- 72,082 mature restaurants for analysis

---

## Features

The extraction script generates **128+ features** including:
- Shannon Entropy (multi-scale)
- POI Counts & Densities
- Competition Metrics
- Demographics
- Indonesia-Specific (mosque, pasar, etc.)
- Temporal multipliers

See `docs/PROJECT_STRUCTURE.md` for full feature list.

---

## Documentation

- `docs/PROJECT_STRUCTURE.md` - Complete structure and workflow
- `docs/README_CLEAN.md` - Detailed quick start
- `docs/00_START_HERE.md` - Original project intro

---

## Workflow

1. **Feature Extraction** → `extract_features.py`
2. **Model Training** → `notebooks/kaggle_survival_training_advanced.ipynb`
3. **Analysis** → `notebooks/kaggle_feature_importance_analysis.ipynb`

---

## Requirements

- Python 3.8+
- pandas, geopandas, numpy
- scikit-survival
- shapely, tqdm

Install: `pip install -r requirements.txt`

---

**Status:** Production Ready ✅
**Last Updated:** 2025-11-19
"""

readme_path = BASE_DIR / 'README.md'
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"  ✓ Created: README.md\n")

# ============================================================================
# STEP 6: Rename main script to simpler name
# ============================================================================

print("Renaming main script...")

old_name = BASE_DIR / 'kaggle_feature_extraction_with_checkpoints.py'
new_name = BASE_DIR / 'extract_features.py'

if old_name.exists():
    shutil.move(str(old_name), str(new_name))
    print(f"  ✓ kaggle_feature_extraction_with_checkpoints.py → extract_features.py\n")

# ============================================================================
# STEP 7: Final summary
# ============================================================================

print("="*80)
print("CLEANUP COMPLETE!")
print("="*80)
print()
print("Root directory now contains:")
print()

root_files = sorted([f.name for f in BASE_DIR.iterdir() if f.is_file()])
for f in root_files:
    if f != 'aggressive_cleanup.py':  # Don't show self
        print(f"  ✓ {f}")

print()
print("Organized directories:")
print()
print(f"  📁 notebooks/  ({len(list((BASE_DIR / 'notebooks').glob('*.ipynb')))} files)")
print(f"  📁 docs/       ({len(list((BASE_DIR / 'docs').glob('*.md')))} files)")
print(f"  📁 scripts/    ({len(list((BASE_DIR / 'scripts').glob('*.py')))} files)")
print(f"  📁 src/        (source modules)")
print(f"  📁 data/       (input data)")
print(f"  📁 outputs/    (generated files)")
print(f"  📁 archive/    (old files)")
print()
print("="*80)
print()
print("Next steps:")
print("  1. python extract_features.py")
print("  2. jupyter notebook notebooks/kaggle_survival_training_advanced.ipynb")
print()
print("="*80)

# Move self to scripts/ as final step
print("\nMoving cleanup script to scripts/...")
self_dst = BASE_DIR / 'scripts' / 'aggressive_cleanup.py'
shutil.copy(__file__, str(self_dst))
print("  ✓ Copied aggressive_cleanup.py → scripts/")
print("\nYou can now delete: aggressive_cleanup.py from root")
