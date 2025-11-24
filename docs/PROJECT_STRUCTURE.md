# POI Survival Analysis Project - Directory Structure

## Overview
This project analyzes Point of Interest (POI) survival in Jakarta using spatial features and survival analysis models.

---

## Current Directory Structure

```
POI/
│
├── 📄 Main Scripts & Notebooks (Root)
│   ├── kaggle_feature_extraction_with_checkpoints.py    ⭐ MAIN FEATURE EXTRACTION SCRIPT
│   ├── kaggle_feature_extraction_complete.ipynb         Main feature extraction notebook
│   ├── kaggle_survival_training_advanced.ipynb          Model training notebook
│   ├── kaggle_feature_importance_analysis.ipynb         Feature analysis notebook
│   ├── requirements.txt                                  Python dependencies
│   └── cleanup_and_organize.py                          Directory cleanup utility
│
├── 📁 src/                                              Source Code Modules
│   ├── data/                                             Data collection & loading
│   │   ├── collect_bps.py
│   │   ├── collect_foursquare.py
│   │   ├── collect_osm.py
│   │   ├── data_loader.py
│   │   └── survival_labeler.py
│   ├── features/                                         Feature engineering
│   │   ├── feature_engineer.py
│   │   └── spatial_features.py
│   ├── models/                                           Model training
│   │   ├── survival_trainer.py
│   │   └── train_model.py
│   └── utils/                                            Utilities
│       ├── config_loader.py
│       └── experiment_tracker.py
│
├── 📁 data/processed/                                   Raw Input Data
│   ├── foursquare/
│   │   └── jakarta_pois_foursquare_iceberg.csv
│   ├── osm/
│   │   └── jakarta_pois_osm.csv
│   ├── buildings/
│   │   └── jakarta_buildings_osm.csv
│   └── bps/
│       ├── jakarta_regencies.csv
│       └── provinces.csv
│
├── 📁 outputs/                                          Generated Outputs
│   ├── kaggle_clean_data/
│   │   └── jakarta_clean_categorized.csv                ⭐ MAIN CLEANED DATASET (27MB)
│   ├── kaggle_raw_data/
│   │   └── jakarta_selatan_raw.csv
│   ├── features/                                         (Will contain feature outputs)
│   └── archive/                                          Old/intermediate outputs
│       ├── jakarta_restaurant_phase1_demographics.csv
│       ├── jakarta_restaurant_phase1_2_5_combined.csv
│       ├── coffee_shops_with_features.csv
│       ├── feature_importance.csv
│       ├── survival_analysis/
│       ├── survival_analysis_jaksel/
│       └── survival_analysis_jaksel_fast/
│
├── 📁 archive/                                          Archived Files
│   ├── notebooks/                                        Old/experimental notebooks (11 files)
│   │   ├── kaggle_phase1_demographics.ipynb
│   │   ├── kaggle_phase2_competition.ipynb
│   │   ├── kaggle_phase3_accessibility.ipynb
│   │   ├── kaggle_phase4_indonesia_specific.ipynb
│   │   ├── kaggle_phases_all_in_one.ipynb
│   │   └── ... (6 more)
│   └── scripts/                                          Old data collection scripts (18 files)
│       ├── collect_boundaries.py
│       ├── collect_buildings.py
│       ├── create_clean_categorized_dataset.py
│       └── ... (15 more)
│
└── 📁 notebooks/                                        Original Exploratory Notebooks
    ├── 01_data_collection.ipynb
    ├── 01_exploratory_data_analysis.ipynb
    └── 04_model_training.ipynb
```

---

## Key Files

### 🎯 Production Files (What You Need)

1. **kaggle_feature_extraction_with_checkpoints.py** ⭐
   - Complete feature extraction pipeline
   - Saves checkpoints after each section
   - Works on both Kaggle and local
   - Generates 128+ features

2. **jakarta_clean_categorized.csv** ⭐
   - Main input dataset (158,377 POIs)
   - Located in: `outputs/kaggle_clean_data/`
   - Contains: 77,918 restaurants

3. **kaggle_survival_training_advanced.ipynb**
   - Model training notebook
   - Uses extracted features for survival prediction

4. **kaggle_feature_importance_analysis.ipynb**
   - Analyzes which features are most important
   - Feature importance visualization

---

## Workflow

### Step 1: Feature Extraction
```bash
python kaggle_feature_extraction_with_checkpoints.py
```

**Output:**
- `outputs/features/jakarta_restaurant_features_complete.csv` (Final dataset with 128+ features)
- `outputs/features/checkpoint_*.csv` (9 checkpoint files)
- `outputs/features/feature_list_complete.txt` (Feature documentation)

### Step 2: Model Training
Open and run: `kaggle_survival_training_advanced.ipynb`

### Step 3: Analysis
Open and run: `kaggle_feature_importance_analysis.ipynb`

---

## Feature Groups (128+ Features)

Generated by the feature extraction script:

| Group | Count | Examples |
|-------|-------|----------|
| Shannon Entropy | 3 | `entropy_500m`, `entropy_1000m`, `entropy_2000m` |
| POI Counts | 48 | `competitors_count_500m`, `mall_count_1000m` |
| POI Densities | 49 | `competitors_density_500m`, `office_density_1000m` |
| Distances | 8 | `nearest_competitor_m`, `dist_city_center_km` |
| Competition | 3 | `avg_competitor_dist_2km`, `cannibalization_risk_500m` |
| Demographics | 3 | `income_district_m`, `density_district`, `working_age_district` |
| Accessibility | 3 | `dist_city_center_km`, `transport_density_1km`, `urban_centrality` |
| Interactions | 6 | `income_pop_interaction`, `office_transport`, `demand_supply_ratio` |
| Indonesia-Specific | 36 | `mosque_count_500m`, `pasar_proximity_score`, `friday_prayer_impact` |
| Temporal | 5 | `ramadan_evening_multiplier`, `gajian_multiplier` |

**Total: 128 features**

---

## Cleanup Summary

### Files Archived
- ✅ 11 old/experimental notebooks → `archive/notebooks/`
- ✅ 18 old data collection scripts → `archive/scripts/`
- ✅ 7 intermediate output files → `outputs/archive/`

### Files Kept
- ✅ 4 essential notebooks (feature extraction, training, analysis)
- ✅ 1 main production script (with checkpoints)
- ✅ Source code modules (`src/`)
- ✅ Main dataset (27MB)
- ✅ Requirements.txt

---

## Next Steps

1. **Run Feature Extraction:**
   ```bash
   python kaggle_feature_extraction_with_checkpoints.py
   ```
   Expected runtime: ~60-90 minutes
   Output: `outputs/features/jakarta_restaurant_features_complete.csv`

2. **Upload to Kaggle:**
   - Upload the generated feature CSV to Kaggle dataset
   - Use in training notebook

3. **Train Models:**
   - Open `kaggle_survival_training_advanced.ipynb`
   - Run all cells to train survival models

4. **Analyze Results:**
   - Open `kaggle_feature_importance_analysis.ipynb`
   - Identify top features

---

## Data Flow

```
data/processed/             →    outputs/kaggle_clean_data/    →    outputs/features/
(Raw POI data)                   (Cleaned & categorized)            (Extracted features)
                                 jakarta_clean_categorized.csv      jakarta_restaurant_features_complete.csv
                                 158,377 POIs                       72,082 mature restaurants
                                                                    128+ features each
```

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Main packages:
- pandas
- geopandas
- numpy
- scikit-survival
- shapely
- tqdm

---

## Notes

- All archived files are safe to delete if needed (kept for reference)
- Main dataset is in `outputs/kaggle_clean_data/`
- Feature extraction saves checkpoints automatically
- Both Kaggle and local environments supported

---

**Last Updated:** 2025-11-19
**Project:** POI Survival Analysis - Jakarta Restaurants
