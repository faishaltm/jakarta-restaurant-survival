# POI Survival Analysis - Jakarta Restaurants

**Clean and Organized Project Structure**

---

## 🎯 Quick Start

### 1. Extract Features (60-90 minutes)
```bash
python kaggle_feature_extraction_with_checkpoints.py
```

### 2. Train Model
Open and run: `kaggle_survival_training_advanced.ipynb`

### 3. Analyze Results
Open and run: `kaggle_feature_importance_analysis.ipynb`

---

## 📁 Project Structure

### **Root Directory** (Essential Files Only)

```
POI/
├── kaggle_feature_extraction_with_checkpoints.py    ⭐ MAIN SCRIPT
├── kaggle_feature_extraction_complete.ipynb         Feature extraction notebook
├── kaggle_survival_training_advanced.ipynb          Model training
├── kaggle_feature_importance_analysis.ipynb         Analysis
├── requirements.txt                                  Dependencies
├── PROJECT_STRUCTURE.md                             Full structure docs
└── cleanup_and_organize.py                          Cleanup utility
```

### **Key Directories**

- **`outputs/kaggle_clean_data/`** - Main dataset (jakarta_clean_categorized.csv - 27MB)
- **`outputs/features/`** - Generated features (created by extraction script)
- **`src/`** - Source code modules
- **`data/processed/`** - Raw input data
- **`archive/`** - Old notebooks and scripts (29 files archived)

---

## 🔥 Main Files

| File | Purpose | Size |
|------|---------|------|
| `kaggle_feature_extraction_with_checkpoints.py` | Feature extraction with auto-save | 29KB |
| `outputs/kaggle_clean_data/jakarta_clean_categorized.csv` | Main dataset | 27MB |
| `kaggle_survival_training_advanced.ipynb` | Model training | 35KB |
| `kaggle_feature_importance_analysis.ipynb` | Feature analysis | 24KB |

---

## 📊 Dataset Overview

**Main Dataset:** `outputs/kaggle_clean_data/jakarta_clean_categorized.csv`

- **Total POIs:** 158,377
- **Restaurants:** 77,918
- **Mature Restaurants:** 72,082 (for analysis)
  - Failures: 3,934
  - Successes: 68,148

---

## ✨ Features Generated (128+ Features)

The extraction script generates:

| Category | Count | Examples |
|----------|-------|----------|
| Shannon Entropy | 3 | Multi-scale diversity (500m, 1km, 2km) |
| POI Counts | 48 | Competitors, malls, offices, etc. |
| POI Densities | 49 | Per km² density calculations |
| Distances | 8 | Nearest competitors, city center |
| Competition Metrics | 3 | Cannibalization risk, avg distance |
| Demographics | 3 | Income, population density |
| Accessibility | 3 | City center distance, transport |
| Interactions | 6 | Income×pop, office×transport |
| Indonesia-Specific | 36 | Mosque, pasar, Friday prayer impact |
| Temporal | 5 | Ramadan, gajian multipliers |

**Total:** 128 features

---

## 🚀 Workflow

### Step 1: Feature Extraction

```bash
python kaggle_feature_extraction_with_checkpoints.py
```

**What it does:**
- Loads jakarta_clean_categorized.csv (27MB)
- Extracts 128+ features in 9 sections
- Saves checkpoint after each section
- Creates final dataset: `jakarta_restaurant_features_complete.csv`

**Runtime:** 60-90 minutes

**Output files:**
```
outputs/features/
├── jakarta_restaurant_features_complete.csv    (Final dataset)
├── feature_list_complete.txt                   (Feature documentation)
├── checkpoint_section1_entropy.csv
├── checkpoint_section2_poi_features.csv
├── ... (9 checkpoints total)
```

### Step 2: Model Training

Open: `kaggle_survival_training_advanced.ipynb`

**What it does:**
- Loads extracted features
- Trains survival models (RSF, GBM, Cox)
- Evaluates performance (C-index)
- Saves trained models

### Step 3: Feature Analysis

Open: `kaggle_feature_importance_analysis.ipynb`

**What it does:**
- Analyzes feature importance
- Creates visualizations
- Identifies top predictive features

---

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

**Main packages:**
- pandas
- geopandas
- numpy
- scikit-survival
- shapely
- tqdm

---

## 🧹 Cleanup Summary

**Archived (moved to `archive/`):**
- 11 old experimental notebooks
- 18 old data collection scripts
- 7 intermediate output files

**Kept (in root):**
- 4 essential notebooks
- 1 main production script
- Source code modules
- Main dataset

**You can safely delete `archive/` if needed** (kept for reference only)

---

## 📖 Documentation

- **`PROJECT_STRUCTURE.md`** - Complete directory structure and workflow
- **`00_START_HERE.md`** - Original project introduction
- **`README_COMPLETE_WORKFLOW.md`** - Detailed workflow guide

---

## 🎯 For Kaggle Users

### Upload to Kaggle:

1. **Upload Script:**
   - Create new Kaggle notebook
   - Copy contents of `kaggle_feature_extraction_with_checkpoints.py`
   - Or upload as Python script

2. **Upload Dataset:**
   - Upload `outputs/kaggle_clean_data/jakarta_clean_categorized.csv`
   - Use as input dataset

3. **Run:**
   - Execute feature extraction
   - Download outputs from `/kaggle/working/`

4. **Train:**
   - Upload feature CSV as new dataset
   - Run `kaggle_survival_training_advanced.ipynb`

---

## 🔍 Troubleshooting

### Issue: Script fails to save files

**Kaggle:**
- Files automatically save to `/kaggle/working/`
- Check "Output" tab for generated files

**Local:**
- Check that `outputs/features/` directory exists
- Script auto-creates it if missing

### Issue: Out of memory

**Solution:**
- Use Kaggle GPU environment (16GB RAM)
- Or reduce buffer sizes in config section

### Issue: Unicode errors

**Solution:**
- Already fixed in the script (Windows encoding)
- Checkpoints save correctly

---

## 📊 Expected Results

After running the complete pipeline:

- **Feature Extraction:** 72,082 restaurants × 128+ features
- **Model Performance:** C-index ~0.75-0.80
- **Top Features:** Shannon entropy, competitor density, demographics

---

## 🤝 Contributing

This is a research project. To contribute:

1. Keep archived files as reference
2. Use `kaggle_feature_extraction_with_checkpoints.py` as main script
3. Document new features in PROJECT_STRUCTURE.md
4. Run cleanup after major changes

---

## 📝 Notes

- **Dataset:** Jakarta restaurants from Foursquare & OSM
- **Analysis:** Survival analysis (failure prediction)
- **Features:** Spatial, demographic, competition, Indonesia-specific
- **Models:** Random Survival Forest, Gradient Boosting, Cox PH

---

**Last Updated:** 2025-11-19
**Status:** Production Ready ✅
**Cleanup:** Complete ✅

---

For detailed documentation, see `PROJECT_STRUCTURE.md`
