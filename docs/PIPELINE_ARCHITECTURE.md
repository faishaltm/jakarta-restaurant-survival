# Pipeline Architecture

## 🏗️ Modular Pipeline Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      run_pipeline.py (CLI)                      │
│                 Main orchestrator with argparse                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
┌───────────────────▼──────────────┐  ┌──────▼──────────────────┐
│   config/pipeline_config.yaml    │  │  src/utils/             │
│   ────────────────────────────   │  │  config_loader.py       │
│   • Geographic settings          │◄─┤  ───────────────────    │
│   • POI categories               │  │  • Load YAML config     │
│   • Feature engineering params   │  │  • Validate settings    │
│   • Model configurations         │  │  • Get/set values       │
│   • Output settings              │  │                         │
└──────────────────────────────────┘  └─────────────────────────┘
                    │
        ┌───────────┼───────────┬───────────────┐
        │           │           │               │
┌───────▼──────┐ ┌──▼────────┐ ┌▼──────────┐ ┌─▼────────────┐
│ STEP 1:      │ │ STEP 2:   │ │ STEP 3:   │ │ STEP 4:      │
│ Data Loading │─► Labeling  │─► Features  │─► Training     │
└──────────────┘ └───────────┘ └───────────┘ └──────────────┘
```

---

## 📦 Module Breakdown

### 1. ConfigLoader (`src/utils/config_loader.py`)
```python
config = ConfigLoader()
config.get('geographic.bbox.min_lat')        # Get value
config.get_buffer_distances()                # Get list
config.update('model.test_size', 0.25)       # Update
config.save('config/experiment_1.yaml')      # Save new config
```

**Responsibilities:**
- Load & validate YAML configuration
- Provide easy access to nested config values
- Support config updates for experiments

---

### 2. DataLoader (`src/data/data_loader.py`)
```python
loader = DataLoader(config)

# Load individual datasets
gdf_fsq = loader.load_foursquare_pois()
gdf_osm = loader.load_osm_pois()
gdf_buildings = loader.load_buildings()

# Load all at once
data = loader.load_all(sample_foursquare=10000)

# Filter & categorize
gdf_coffee = loader.filter_coffee_shops(gdf_fsq)
gdf_categorized = loader.categorize_pois(gdf_fsq)
```

**Responsibilities:**
- Load Foursquare, OSM, buildings, boundaries, population
- Parse category arrays
- Filter coffee shops
- Categorize POIs (university, office, mall, etc.)
- Support sampling for testing

---

### 3. FeatureEngineer (`src/features/feature_engineer.py`)
```python
engineer = FeatureEngineer(config)

# Create individual feature types
gdf = engineer.create_proximity_features(target_gdf, reference_gdfs)
gdf = engineer.create_density_features(target_gdf, reference_gdfs)
gdf = engineer.create_competitor_features(target_gdf, coffee_shops)
gdf = engineer.create_diversity_features(target_gdf, all_pois)

# Or create all features at once
gdf_with_features = engineer.create_all_features(
    target_gdf,
    data,
    population_raster
)

# Get list of created features
feature_names = engineer.get_feature_names()
```

**Feature Types:**

1. **Proximity Features** (Distance to nearest)
   - `dist_nearest_university`
   - `dist_nearest_office`
   - `dist_nearest_mall`
   - `dist_nearest_transport`
   - etc.

2. **Density Features** (Count within buffer)
   - `count_university_500m`
   - `count_office_1000m`
   - `count_competitors_500m`
   - `count_buildings_1000m`
   - etc.

3. **Competitor Features**
   - `dist_nearest_competitor`
   - `count_competitors_150m`
   - `count_competitors_500m`
   - etc.

4. **Diversity Features** (POI variety)
   - `poi_diversity_shannon_500m`
   - `poi_diversity_simpson_1000m`

5. **Population Features**
   - `population_density` (from WorldPop raster)

**Spatial Methods:**
- Uses `cKDTree` for efficient spatial queries
- Buffers: 150m, 500m, 1km, 2km, 5km (configurable)
- Metric CRS (UTM 48S) for accurate distance calculations

---

### 4. ModelTrainer (`src/models/model_trainer.py`)
```python
trainer = ModelTrainer(config)

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_training_data(df)

# Train individual models
rf_model = trainer.train_random_forest(X_train, y_train)
xgb_model = trainer.train_xgboost(X_train, y_train)

# Or train all enabled models with tuning
results = trainer.train_all_models(
    X_train, X_test, y_train, y_test,
    tune=True
)

# Feature importance
importance_df = trainer.get_feature_importance()

# Save model
trainer.save_model(model_name='random_forest', output_dir='models')
```

**Models Supported:**
- Random Forest (sklearn)
- XGBoost
- LightGBM (optional)

**Hyperparameter Tuning:**
- GridSearchCV
- RandomizedSearchCV
- Cross-validation (5-fold default)

**Evaluation Metrics:**
- Accuracy
- F1 Score (weighted)
- Precision/Recall
- Confusion Matrix
- Feature Importance

---

## 🔄 Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUT DATA (data/processed/)                                    │
├──────────────────────────────────────────────────────────────────┤
│ • foursquare/jakarta_pois_foursquare_iceberg.csv  (2.5M POIs)  │
│ • osm/jakarta_pois_osm.geojson                    (5K POIs)     │
│ • buildings/jakarta_buildings_osm.geojson         (5.6K)        │
│ • boundaries/indonesia_adm3.geojson               (districts)   │
│ • population/indonesia_pop_density_2020_1km.tif   (raster)      │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA LOADING                                            │
├──────────────────────────────────────────────────────────────────┤
│ DataLoader.load_all()                                           │
│ • Parse category arrays                                          │
│ • Categorize POIs (university, office, mall, etc.)              │
│ • Filter coffee shops                                            │
│ • Extract POI types                                              │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: LABELING                                                │
├──────────────────────────────────────────────────────────────────┤
│ • Parse date_closed column                                       │
│ • Label: 1 = operating (success), 0 = closed (failure)          │
│ • Output: coffee_shops with 'label' column                      │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: FEATURE ENGINEERING                                     │
├──────────────────────────────────────────────────────────────────┤
│ FeatureEngineer.create_all_features()                           │
│                                                                  │
│ For each coffee shop:                                           │
│ ┌────────────────────────────────────────────────────┐         │
│ │ Proximity: dist_nearest_university, office, mall   │         │
│ │ Density:   count_office_500m, mall_1000m           │         │
│ │ Competitor: count_competitors_500m                 │         │
│ │ Diversity: shannon_entropy_1000m                   │         │
│ │ Population: extract from raster                    │         │
│ │ Buildings: count_buildings_500m                    │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                  │
│ Output: coffee_shops with ~50-150 feature columns               │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: MODEL TRAINING                                          │
├──────────────────────────────────────────────────────────────────┤
│ ModelTrainer.train_all_models()                                 │
│                                                                  │
│ 1. Train-test split (70:30)                                     │
│ 2. Hyperparameter tuning (GridSearch/RandomSearch)             │
│ 3. Train models (Random Forest, XGBoost)                        │
│ 4. Evaluate (accuracy, F1, precision, recall)                   │
│ 5. Feature importance (SHAP)                                     │
│ 6. Save best model                                               │
│                                                                  │
│ Output: Trained model + predictions + feature importance        │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ OUTPUT FILES                                                     │
├──────────────────────────────────────────────────────────────────┤
│ • outputs/features/coffee_shops_with_features.csv               │
│ • outputs/results/feature_importance.csv                        │
│ • models/random_forest_model.pkl                                │
│ • models/xgboost_model.pkl                                       │
│ • models/*_features.json                                         │
│ • logs/pipeline_YYYY-MM-DD.log                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration-Driven Design

All parameters externalized to `config/pipeline_config.yaml`:

```yaml
# Example: Change buffer distances
feature_engineering:
  buffer_distances_meters: [150, 500, 1000, 2000]  # Modify here

# Example: Enable/disable features
feature_engineering:
  density_features:
    enabled: true                    # Toggle on/off
    poi_types: [university, office]  # Select which POI types

# Example: Switch models
model:
  models:
    random_forest:
      enabled: false                 # Disable RF
    xgboost:
      enabled: true                  # Enable XGBoost
```

**Benefits:**
- No code changes needed for experiments
- Easy to version control experiments
- Reproducible results

---

## 🧪 Testing Strategy

### Unit Tests
Each module has `__main__` block for standalone testing:

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

### Integration Test
```bash
# Quick pipeline test (5 min)
python run_pipeline.py --sample 5000 --no-tune --log-level DEBUG
```

### Full Pipeline
```bash
# Production run
python run_pipeline.py
```

---

## 📊 Performance Optimization

### Memory Efficiency
- **Optimized dtypes**: Use `float32` instead of `float64`
- **Sampling support**: Test with subset before full run
- **Streaming**: Load data in chunks if needed

### Computation Speed
- **Spatial indexing**: Use `cKDTree` for O(log n) queries
- **Parallel processing**: Models use `n_jobs=-1`
- **Caching**: Config loaded once, reused across modules

### Scalability
```python
# From 10K to 2.5M POIs seamlessly
loader.load_all(sample_foursquare=10000)   # Testing
loader.load_all(sample_foursquare=None)    # Production
```

---

## 🔄 Iteration Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Quick Test (--sample 10000 --no-tune)             │
│     • 5 minutes                                         │
│     • Verify pipeline works                             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  2. Analyze Feature Importance                          │
│     • cat outputs/results/feature_importance.csv        │
│     • Identify top features                             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  3. Adjust Configuration                                │
│     • Edit config/pipeline_config.yaml                  │
│     • Remove low-importance features                    │
│     • Tune hyperparameters                              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  4. Re-run with Optimized Config                        │
│     • python run_pipeline.py --config optimized.yaml    │
│     • Compare results                                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  5. Scale to Full Dataset                               │
│     • python run_pipeline.py                            │
│     • Final model with all 2.5M POIs                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Design Principles

1. **Modularity**: Each component (loader, engineer, trainer) is independent
2. **Configuration-Driven**: All parameters externalized
3. **Reusability**: Modules can be imported and used separately
4. **Testability**: Each module has standalone test capability
5. **Scalability**: Works with 10K or 2.5M POIs seamlessly
6. **Reproducibility**: Config files ensure repeatable experiments
7. **Extensibility**: Easy to add new features, models, or data sources

---

**Architecture designed for rapid experimentation and production deployment** 🚀
