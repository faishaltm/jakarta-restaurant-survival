# Jakarta Coffee Shop Site Selection - Modular Pipeline

**Version**: 1.0.0

Modular, config-driven ML pipeline untuk prediksi lokasi optimal coffee shop di Jakarta menggunakan 2.5M+ POI data.

---

## 🎯 Features

### ✅ Modular Architecture
- **Config-driven**: Semua parameter di `config/pipeline_config.yaml`
- **Reusable modules**: DataLoader, FeatureEngineer, ModelTrainer
- **Easy experimentation**: Ubah config tanpa edit code

### ✅ Comprehensive Feature Engineering
Berdasarkan best practices dari research papers:
- **Proximity features**: Distance to universities, offices, malls, transport
- **Density features**: Count POIs within 150m/500m/1km/2km/5km buffers
- **Competitor analysis**: Nearby coffee shop counts
- **Diversity metrics**: Shannon entropy & Simpson index
- **Population density**: From WorldPop raster
- **Building density**: OSM building counts

### ✅ Multiple ML Models
- Random Forest (baseline)
- XGBoost (high performance)
- LightGBM (optional)
- Automatic hyperparameter tuning (GridSearch/RandomSearch)
- SHAP feature importance

### ✅ Easy CLI Interface
```bash
# Run full pipeline
python run_pipeline.py

# Quick test with sample
python run_pipeline.py --sample 10000 --no-tune

# Debug mode
python run_pipeline.py --log-level DEBUG --sample 5000
```

---

## 📁 Project Structure

```
POI/
├── config/
│   └── pipeline_config.yaml          # All configuration parameters
│
├── src/
│   ├── utils/
│   │   └── config_loader.py          # Config management
│   ├── data/
│   │   └── data_loader.py            # Data loading & preprocessing
│   ├── features/
│   │   └── feature_engineer.py       # Feature engineering
│   └── models/
│       └── model_trainer.py          # Model training & tuning
│
├── data/
│   └── processed/                     # Input datasets (2.5M POIs)
│
├── outputs/
│   ├── features/                      # Engineered features
│   └── results/                       # Model results & predictions
│
├── models/                            # Saved trained models
├── logs/                              # Pipeline logs
│
├── run_pipeline.py                    # Main CLI script
├── requirements.txt                   # Dependencies
└── PIPELINE_README.md                 # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Configure Pipeline
Edit `config/pipeline_config.yaml`:

```yaml
# Example: Quick test with reduced features
feature_engineering:
  buffer_distances_meters:
    - 500
    - 1000

  density_features:
    enabled: true
    poi_types:
      - university
      - office
      - mall

model:
  models:
    random_forest:
      enabled: true
    xgboost:
      enabled: false  # Disable for faster testing
```

### 3. Run Pipeline

**Option A: Quick Test (Recommended First)**
```bash
# Test with 10K POIs, no tuning (fast: ~5 minutes)
python run_pipeline.py --sample 10000 --no-tune
```

**Option B: Full Pipeline**
```bash
# Use all 2.5M POIs with hyperparameter tuning (slow: ~2-4 hours)
python run_pipeline.py
```

**Option C: Custom Config**
```bash
# Use custom configuration file
python run_pipeline.py --config my_experiment.yaml --sample 50000
```

### 4. Check Results
```bash
# Feature importance
cat outputs/results/feature_importance.csv

# Trained model
ls models/

# Logs
tail -f logs/pipeline_*.log
```

---

## ⚙️ Configuration Guide

### Key Configuration Sections

#### 1. Geographic Settings
```yaml
geographic:
  bbox:
    min_lat: -6.4
    max_lat: -6.0
    min_lon: 106.6
    max_lon: 107.1

  grid_size_meters: 100        # For grid-based analysis
  analysis_window_meters: 1000  # Feature extraction radius
```

#### 2. POI Categories
```yaml
poi_categories:
  coffee_keywords:
    - coffee
    - café
    - kopi

  poi_types:
    university: [university, college, kampus]
    office: [office, kantor, coworking]
    mall: [mall, shopping center, plaza]
    # ... etc
```

#### 3. Feature Engineering
```yaml
feature_engineering:
  # Buffer distances (from literature: 150m, 500m, 1km, 2km)
  buffer_distances_meters: [150, 500, 1000, 2000, 5000]

  # Enable/disable feature types
  proximity_features:
    enabled: true
    poi_types: [university, office, mall, transport]

  density_features:
    enabled: true
    include_competitors: true
    include_buildings: true

  diversity_features:
    enabled: true

  population_features:
    enabled: true
```

#### 4. Model Configuration
```yaml
model:
  test_size: 0.3
  stratify: true

  models:
    random_forest:
      enabled: true
      hyperparameters:
        n_estimators: [100, 200, 300]
        max_depth: [10, 20, 30]

    xgboost:
      enabled: true
      hyperparameters:
        n_estimators: [100, 200, 300]
        learning_rate: [0.01, 0.05, 0.1]

  tuning:
    method: "grid_search"  # or "random_search"
    cv_folds: 5
    scoring: "f1"
```

#### 5. Labeling Strategy
```yaml
labeling:
  success_criteria:
    use_date_closed: true  # Use closure status as label
    min_operation_months: 12
```

---

## 🔧 Customization & Experimentation

### Experiment 1: Feature Selection
Test different feature combinations:

```yaml
# Experiment: Only proximity features
feature_engineering:
  proximity_features:
    enabled: true
  density_features:
    enabled: false
  diversity_features:
    enabled: false
```

Run:
```bash
python run_pipeline.py --config config/exp_proximity_only.yaml
```

### Experiment 2: Different Buffer Distances
Based on literature findings (81% shops within 150m of main roads):

```yaml
feature_engineering:
  buffer_distances_meters:
    - 150   # Main road proximity
    - 500   # Immediate neighborhood
    - 1000  # Walking distance
```

### Experiment 3: Model Comparison
```yaml
model:
  models:
    random_forest:
      enabled: true
    xgboost:
      enabled: true
    lightgbm:
      enabled: true

  tuning:
    method: "random_search"  # Faster than grid_search
```

---

## 📊 Output Files

### 1. Features
- `outputs/features/coffee_shops_with_features.csv` - All engineered features

### 2. Model Results
- `outputs/results/feature_importance.csv` - Feature importance ranking
- `models/random_forest_model.pkl` - Trained Random Forest
- `models/xgboost_model.pkl` - Trained XGBoost
- `models/*_features.json` - Feature names used in training

### 3. Logs
- `logs/pipeline_YYYY-MM-DD.log` - Detailed execution logs

---

## 🧪 Testing & Validation

### Unit Test Individual Modules
```bash
# Test config loader
python src/utils/config_loader.py

# Test data loader
python src/data/data_loader.py

# Test feature engineer
python src/features/feature_engineer.py

# Test model trainer
python src/models/model_trainer.py
```

### Quick Pipeline Test
```bash
# Minimal test (5 minutes)
python run_pipeline.py \
  --sample 5000 \
  --no-tune \
  --log-level DEBUG
```

---

## 📈 Performance Benchmarks

### Processing Time Estimates

| Configuration | Foursquare Sample | Tuning | Time | Features |
|--------------|------------------|--------|------|----------|
| **Quick Test** | 10,000 | No | ~5 min | ~50 |
| **Medium** | 100,000 | No | ~20 min | ~100 |
| **Full (No Tune)** | 2,553,079 | No | ~45 min | ~150 |
| **Full (With Tune)** | 2,553,079 | Yes | ~2-4 hrs | ~150 |

### Memory Usage
- **10K POIs**: ~500 MB
- **100K POIs**: ~2 GB
- **2.5M POIs**: ~8-12 GB

---

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Use sampling
```bash
python run_pipeline.py --sample 100000
```

### Issue: Slow Feature Engineering
**Solution**: Reduce buffer distances
```yaml
buffer_distances_meters: [500, 1000]  # Instead of [150, 500, 1000, 2000, 5000]
```

### Issue: Hyperparameter Tuning Too Slow
**Solution**: Use random search or disable tuning
```bash
python run_pipeline.py --no-tune
```

Or in config:
```yaml
model:
  tuning:
    method: "random_search"  # Faster than grid_search
```

---

## 📚 Research References

Pipeline based on best practices from:

1. **Beijing Coffee Shop Study (MDPI 2023)**
   - 23 factors from 20 POI types
   - 150m buffer from main roads (81% concentration)
   - Random Forest: R²=0.929

2. **Luckin/Starbucks Shanghai Study (MDPI 2025)**
   - 100m x 100m grid analysis
   - 1000m radius feature window
   - RF accuracy: 90-92%
   - SHAP interpretation

3. **Spatial Feature Engineering (Geographic Data Science)**
   - Proximity, density, diversity features
   - Spatial lag & summary statistics

---

## 🔄 Iteration Workflow

### Recommended Experimentation Process

1. **Baseline** (Quick test)
```bash
python run_pipeline.py --sample 10000 --no-tune
```

2. **Analyze** features
```bash
cat outputs/results/feature_importance.csv | head -20
```

3. **Adjust** config based on feature importance
```yaml
# Remove low-importance features
feature_engineering:
  density_features:
    poi_types:
      - university  # Keep high importance
      - office      # Keep high importance
      # Remove low importance POI types
```

4. **Re-run** with optimized config
```bash
python run_pipeline.py --config config/optimized.yaml
```

5. **Scale up** to full data
```bash
python run_pipeline.py  # No --sample flag
```

---

## 📞 Support

For issues or questions:
1. Check logs: `logs/pipeline_*.log`
2. Enable debug mode: `--log-level DEBUG`
3. Test modules individually (see Testing section)

---

## 🎓 Next Steps

After successful pipeline run:

1. **Analyze Results**
   - Review feature importance
   - Check model performance metrics
   - Validate predictions

2. **Grid-Based Prediction**
   - Create 100m x 100m grid over Jakarta
   - Apply trained model to predict optimal locations
   - Visualize heat map of predicted success

3. **Interactive Dashboard**
   - Build Streamlit/Dash app
   - Allow users to explore predictions
   - What-if analysis tool

4. **Production Deployment**
   - API endpoint for real-time predictions
   - Scheduled retraining pipeline
   - Monitoring & alerting

---

**Happy experimenting! 🚀☕**
