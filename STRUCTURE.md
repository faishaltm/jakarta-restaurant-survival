# Project Directory Structure

📁 **POI - Restaurant Survival Analysis Project**

```
POI/
├── 📄 README.md                    # Project overview
├── 📄 requirements.txt             # Python dependencies
├── 📄 run_setup.bat               # Setup script
│
├── 📁 .claude/                    # Claude Code configuration
│
├── 📁 archive/                    # Old/deprecated files
│
├── 📁 config/                     # Configuration files
│
├── 📁 data/                       # Raw data files
│   ├── jakarta_pois.csv
│   ├── indonesia_population.csv
│   └── ... (other data files)
│
├── 📁 docs/                       # All documentation
│   ├── 00_START_HERE.md          # Quick start guide
│   ├── RANGKUMAN_TRAINING_DAN_RENCANA.md
│   ├── OPTIMIZATION_GUIDE.md
│   ├── KAGGLE_BACKGROUND_EXECUTION_GUIDE.md
│   └── ... (25 documentation files)
│
├── 📁 logs/                       # Training logs
│
├── 📁 models/                     # Trained models
│
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── kaggle_survival_training_advanced.ipynb
│   └── ... (other notebooks)
│
├── 📁 outputs/                    # All outputs
│   ├── 📁 archive/               # Old training outputs
│   ├── 📁 kaggle_clean_data/     # Cleaned data for Kaggle
│   ├── 📁 kaggle_raw_data/       # Raw data for Kaggle
│   └── 📁 visualizations/        # HTML visualizations
│       ├── restaurant_success_vs_failure.html (MAIN)
│       ├── restaurant_with_boundaries.html (NEW)
│       ├── restaurant_comparison_sidebyside.html
│       └── ... (data files and other visualizations)
│
├── 📁 scripts/                    # Python scripts
│   ├── 📁 feature_extraction/
│   │   ├── extract_features.py
│   │   └── extract_features_complete_optimized.py
│   │
│   ├── 📁 visualization/
│   │   ├── create_restaurant_comparison_heatmap.py
│   │   ├── create_contour_comparison_data.py
│   │   ├── create_optimized_heatmap.py
│   │   ├── extract_failure_data.py
│   │   └── optimize_failure_data.py
│   │
│   ├── aggressive_cleanup.py
│   └── cleanup_and_organize.py
│
├── 📁 src/                        # Source code (if any)
│
└── 📁 venv/                       # Virtual environment (excluded from git)
```

---

## 📊 Key Files

### Visualizations (outputs/visualizations/)
- **restaurant_success_vs_failure.html** - Main overlapping heatmap with layer controls
- **restaurant_with_boundaries.html** - Zone-based with circular boundaries (NEW)
- **restaurant_comparison_sidebyside.html** - Split-screen comparison

### Scripts (scripts/)
- **feature_extraction/** - Extract features for model training
- **visualization/** - Generate heatmap visualizations

### Documentation (docs/)
- **00_START_HERE.md** - Quick start guide
- **RANGKUMAN_TRAINING_DAN_RENCANA.md** - Training summary & roadmap
- **OPTIMIZATION_GUIDE.md** - Performance optimization guide
- **KAGGLE_BACKGROUND_EXECUTION_GUIDE.md** - Kaggle execution guide

---

## 🎯 Current Status

**Root Directory:** ✅ Clean (only essential files)
- README.md
- requirements.txt
- run_setup.bat

**All scripts:** ✅ Organized in scripts/ subdirectories
**All documentation:** ✅ Organized in docs/
**All visualizations:** ✅ In outputs/visualizations/

**Total files in root:** 3 (excluding directories)

---

## 📝 Quick Access

### Run Feature Extraction:
```bash
python scripts/feature_extraction/extract_features_complete_optimized.py
```

### Generate Visualizations:
```bash
python scripts/visualization/create_restaurant_comparison_heatmap.py
python scripts/visualization/create_contour_comparison_data.py
```

### View Results:
- Open: `outputs/visualizations/restaurant_success_vs_failure.html`
- Open: `outputs/visualizations/restaurant_with_boundaries.html`
