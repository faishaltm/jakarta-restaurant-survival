# 🍽️ Restaurant Survival Analysis - Jakarta POI

Prediksi keberhasilan restoran menggunakan Survival Analysis dengan data spasial Point of Interest (POI) Jakarta.

**Status**: 🚧 In Development
**Current Phase**: Model Optimization (GBS Testing)
**Best C-index**: 0.7599 (Random Survival Forest)
**Target**: 0.85-0.90

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Dataset** | 72,082 mature restaurants |
| **Failure Rate** | 5.5% (3,934 failures) |
| **Features Extracted** | 130 features |
| **Best Model** | Random Survival Forest (RSF) |
| **Best C-index** | 0.7599 ✅ |
| **Training Time** | ~7-10 minutes (CPU) |

---

## 🎯 Project Goals

1. **Predict restaurant survival** using spatial, demographic, and competition data
2. **Identify key success factors** for restaurant locations in Jakarta
3. **Provide actionable insights** for restaurant entrepreneurs and investors

---

## 📁 Project Structure

```
POI/
├── notebooks/              # Jupyter notebooks for experiments
│   ├── kaggle_gbs_proven_features.ipynb     # [NEW] GBS with proven features
│   ├── kaggle_xgboost_progressive_training.ipynb  # XGBoost experiments
│   └── ...
├── scripts/                # Python scripts
│   ├── feature_extraction/
│   │   └── extract_features_complete_optimized.py  # Feature engineering
│   └── ...
├── docs/                   # Documentation
│   ├── 00_START_HERE.md              # Project overview
│   ├── RANGKUMAN_TRAINING_DAN_RENCANA.md  # Training summary
│   ├── FINDINGS_CORRECTION_Phase4.md      # Research findings
│   └── ...
├── data/                   # Data files (NOT in git - 2.5GB)
│   ├── raw/               # Raw OSM data
│   └── processed/         # Processed features
│       └── features/
│           └── jakarta_restaurant_features_complete.csv  # Main dataset
├── outputs/               # Model outputs (NOT in git - 3.6GB)
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 🚀 Current Progress

### ✅ Completed

1. **Data Collection & Cleaning**
   - ✅ Collected 158,377 POIs from OpenStreetMap
   - ✅ Filtered to 77,918 restaurants
   - ✅ Labeled survival outcomes (72,082 mature restaurants)

2. **Feature Engineering**
   - ✅ Extracted 130 features including:
     - Competition metrics (count, density, distance)
     - Demographics (population, income, working age)
     - Accessibility (transport, distance to center)
     - Indonesia-specific (pasar, mosque, gas stations)
     - Spatial diversity (Shannon entropy)
     - Interaction features
   - ✅ Multiple buffer sizes tested (500m, 1km, 2km, 5km)
   - ✅ Optimal buffer: **5km**

3. **Baseline Models**
   - ✅ Random Survival Forest: **C-index 0.7599** (8 features)
   - ✅ Gradient Boosting: C-index 0.7590 (22 features)
   - ✅ Identified top features:
     1. Competition metrics (80% importance)
     2. Transport access (67%)
     3. Demographics (45-48%)
     4. Traditional markets (pasar)

4. **XGBoost Experiments**
   - ⚠️ XGBoost survival:cox achieved only **C-index 0.41**
   - 🔍 Root cause: Extreme imbalance (17:1) + algorithm sensitivity
   - 📝 Lesson: RSF/GBS better suited for survival analysis

### 🚧 In Progress

5. **Gradient Boosting Survival (GBS) Testing**
   - 📝 Created: `kaggle_gbs_proven_features.ipynb`
   - 🎯 Goal: Match or exceed RSF's 0.76 C-index with faster training
   - 🧪 Testing 4 experiments:
     - Proven 8 features + GBS
     - Proven 8 features + RSF
     - Top 10 features + GBS
     - Top 10 features + RSF
   - ⏳ Status: **Ready for Kaggle testing**

### 📋 Next Steps

6. **Model Optimization**
   - [ ] Run GBS experiments on Kaggle
   - [ ] Compare performance: GBS vs RSF vs XGBoost
   - [ ] Hyperparameter tuning if needed
   - [ ] Select best model configuration

7. **Advanced Features** (if needed to reach 0.85-0.90)
   - [ ] Temporal features (day of week, time-based)
   - [ ] Network analysis (centrality, clustering)
   - [ ] External data integration (reviews, traffic)

8. **Production Deployment**
   - [ ] Model serving API
   - [ ] Web interface for predictions
   - [ ] Documentation and user guide

---

## 🎓 Key Findings

### 🏆 Top Success Factors (From RSF Model)

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | **Competition (5km)** | 80% | Number of competitors is THE most critical factor |
| 2 | **Transport Access** | 67% | Proximity to public transport = higher foot traffic |
| 3 | **Working Age Pop** | 48% | Target demographic > income level! |
| 4 | **Population Density** | 45% | Volume of potential customers matters |
| 5 | **Distance to Center** | 28% | Central locations have advantage |
| 6 | **Traditional Markets** | Variable | Context-dependent (1km vs 5km buffer) |

### 💡 Surprising Insights

1. **Competition is NOT always bad**: Moderate competition (optimal zone) indicates demand
2. **Volume > Wealth**: Working age population 7x more important than district income
3. **Malls are overrated**: Mall proximity has only 0.53% importance (lowest!)
4. **Buffer size matters**: 5km buffer significantly better than 1km/2km

### ❌ Features That Don't Work

- ❌ Hospital count (1.21% importance)
- ❌ School count (1.42% importance)
- ❌ Mall count (0.53% importance - LOWEST!)
- ❌ Income-density interaction (0.00% - redundant)

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- 16GB+ RAM (for feature extraction)
- ~6GB disk space (data + outputs)

### Installation

```bash
# Clone repository
git clone <your-repository-url>
cd POI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Feature extraction (if not done)
python scripts/feature_extraction/extract_features_complete_optimized.py

# 2. Run training notebook
# Upload notebooks/kaggle_gbs_proven_features.ipynb to Kaggle
# Add dataset: jakarta_restaurant_features_complete.csv
# Run all cells

# Expected output: C-index 0.70-0.76 in ~10 minutes
```

---

## 📊 Dataset Information

### Source Data

- **Source**: OpenStreetMap (OSM) Indonesia
- **Collection Date**: November 2024
- **Geographic Scope**: Jakarta (DKI Jakarta)
- **Total POIs**: 158,377 (all categories)
- **Restaurants**: 77,918

### Processed Dataset

**File**: `data/processed/features/jakarta_restaurant_features_complete.csv`

- **Rows**: 72,082 mature restaurants
- **Columns**: 133 (130 features + 3 meta columns)
- **Size**: ~150MB
- **Failure Rate**: 5.5% (3,934 closed, 68,148 active)
- **Imbalance Ratio**: 17.3:1

**Note**: Due to size (2.5GB total), dataset is NOT included in git repository.

---

## 🧪 Experiments Log

### Experiment 1: Thematic Baseline
- **Model**: Random Survival Forest
- **Features**: 8 (competition + demographics + accessibility)
- **Result**: C-index **0.7599** ✅
- **Time**: 7 minutes

### Experiment 2: Feature Importance Analysis
- **Model**: Gradient Boosting (20 trees)
- **Features**: 22 (5 phases)
- **Result**: C-index **0.7590**

### Experiment 3: XGBoost Progressive Training
- **Model**: XGBoost survival:cox
- **Features**: Progressive 8 → 18 → 26 → 29
- **Result**: C-index **0.4147** ❌
- **Lesson**: Stick with survival-specific algorithms (RSF/GBS)

### Experiment 4: GBS with Proven Features [NEXT]
- **Model**: Gradient Boosting Survival
- **Features**: Testing 8 (proven) and 10 (top importance)
- **Expected**: C-index 0.70-0.76
- **Status**: ⏳ Ready for testing

---

## 📚 Documentation

- **[00_START_HERE.md](docs/00_START_HERE.md)**: Project overview
- **[RANGKUMAN_TRAINING_DAN_RENCANA.md](docs/RANGKUMAN_TRAINING_DAN_RENCANA.md)**: Training summary
- **[FINDINGS_CORRECTION_Phase4.md](docs/FINDINGS_CORRECTION_Phase4.md)**: Research findings
- **[STRUCTURE.md](STRUCTURE.md)**: Detailed project structure

---

## 🔄 Git Workflow

### What to Push

✅ **Include**:
- Source code (`scripts/`, `src/`)
- Notebooks (`notebooks/`)
- Documentation (`docs/`, `README.md`)
- Configuration (`config/`, `requirements.txt`)
- `.gitignore`

❌ **Exclude** (in `.gitignore`):
- Data files (`data/` - 2.5GB)
- Outputs (`outputs/` - 3.6GB)
- Models (`models/*.pkl`)
- Virtual environment (`venv/`)
- Cache files (`cache/`, `__pycache__/`)
- Logs (`logs/`)

### First Commit

```bash
# Initialize git (if not done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Restaurant survival analysis project

- Feature extraction: 130 features from 72k restaurants
- Baseline models: RSF (C-index 0.76), GBS experiments
- Documentation: Complete project structure and findings
- Notebooks: XGBoost and GBS experiments ready for testing"

# Add remote and push
git remote add origin <your-github-url>
git branch -M main
git push -u origin main
```

---

## 🤝 Contributing

Contributions welcome for:
- Feature engineering ideas
- Model improvements
- Documentation enhancements
- Bug fixes

---

## 📄 License

[Add your license here]

---

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Create an issue]
- Email: [Your email]

---

## 🙏 Acknowledgments

- **Data Source**: OpenStreetMap contributors
- **Tools**: scikit-survival, XGBoost, GeoPandas
- **Platform**: Kaggle Notebooks

---

**Last Updated**: 2024-11-20
**Version**: 0.4.0 (GBS Testing Phase)
