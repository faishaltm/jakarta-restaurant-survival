# Sequential Workflow: Feature Extraction + Training

**Created**: 2025-11-18
**Purpose**: Two-stage workflow for comprehensive survival prediction

---

## Overview

This workflow separates feature extraction from model training, allowing:
- ✅ **Reusable features**: Extract once, train multiple times
- ✅ **Memory efficient**: No re-computation during experiments
- ✅ **Flexible testing**: Easy to add new features or models
- ✅ **Production ready**: Clear separation of concerns

---

## Workflow Stages

### Stage 1: Feature Extraction
**Notebook**: `kaggle_feature_extraction_complete.ipynb`

**Input**:
- `jakarta_clean_categorized.csv` (raw POI data)

**Output**:
- `jakarta_restaurant_features_complete.csv` (~50-60 features)
- `feature_list_complete.txt` (feature documentation)

**Runtime**: ~30-40 minutes

**What it does**:
1. Load all Jakarta POI data
2. Create survival labels (event_observed, survival_days)
3. Extract 9 feature groups:
   - Shannon Entropy (multi-scale)
   - POI Counts & Densities
   - Indonesia-Specific (mosque, pasar, convenience, gas)
   - Competition Metrics
   - Demographics
   - Accessibility
   - Interactions
   - Indonesia Advanced
   - Temporal

**Key features**:
- Memory optimized for Kaggle T4
- Spatial indexing with STRtree
- Multi-buffer extraction (500m, 1km, 2km, 5km)
- Explicit garbage collection

---

### Stage 2: Model Training
**Notebook**: `kaggle_survival_training_advanced.ipynb`

**Input**:
- `jakarta_restaurant_features_complete.csv` (from Stage 1)

**Output**:
- `feature_importance_all.csv` - Complete feature rankings
- `top_k_feature_results.csv` - Optimal K analysis
- `feature_group_results.csv` - Group performance
- `progressive_feature_results.csv` - Incremental improvement
- `experiment_summary.csv` - Comprehensive results
- 5 visualization PNGs

**Runtime**: ~40-50 minutes

**What it does**:
1. Load pre-extracted features
2. Train baseline models (RSF + GBS)
3. Analyze feature importance
4. Test Top-K selection (5, 10, 15, 20, 30, 40, 50, all)
5. Test each feature group independently
6. Progressive group addition (best to worst)
7. Train final optimized model (300 trees)

**Key experiments**:
- All Features (RSF): Baseline
- All Features (GBS): Alternative algorithm
- Top-K Features: Find optimal number
- Feature Groups: Identify strongest theme
- Progressive Addition: Measure cumulative value
- Final Model: Best configuration with more trees

---

## How to Run

### On Kaggle:

1. **Upload both notebooks**:
   - `kaggle_feature_extraction_complete.ipynb`
   - `kaggle_survival_training_advanced.ipynb`

2. **Add dataset**:
   - `jakarta_clean_categorized.csv`

3. **Select environment**:
   - Accelerator: T4 GPU (recommended for 16GB RAM)
   - Internet: ON (for package installation)

4. **Run Stage 1 (Feature Extraction)**:
   - Open `kaggle_feature_extraction_complete.ipynb`
   - Run all cells
   - Wait ~30-40 minutes
   - Verify output: `jakarta_restaurant_features_complete.csv` created

5. **Run Stage 2 (Training)**:
   - Open `kaggle_survival_training_advanced.ipynb`
   - Run all cells
   - Wait ~40-50 minutes
   - Review results in output CSVs and PNGs

**Total runtime**: ~70-90 minutes

---

### Locally:

1. **Prepare environment**:
   ```bash
   pip install scikit-survival geopandas pandas numpy matplotlib seaborn tqdm
   ```

2. **Prepare data**:
   - Place `jakarta_clean_categorized.csv` in `data/` folder

3. **Run Stage 1**:
   ```bash
   jupyter notebook kaggle_feature_extraction_complete.ipynb
   # Run all cells
   # Output: outputs/feature_extraction/jakarta_restaurant_features_complete.csv
   ```

4. **Run Stage 2**:
   ```bash
   jupyter notebook kaggle_survival_training_advanced.ipynb
   # Run all cells
   # Output: outputs/survival_training_advanced/*.csv and *.png
   ```

---

## File Dependencies

```
jakarta_clean_categorized.csv (INPUT)
         ↓
[Stage 1: Feature Extraction]
         ↓
jakarta_restaurant_features_complete.csv (INTERMEDIATE)
         ↓
[Stage 2: Model Training]
         ↓
Results: 5 CSVs + 5 PNGs (OUTPUT)
```

---

## Feature Count Summary

### Stage 1 Output (~50-60 features):

| Group | Count | Examples |
|-------|-------|----------|
| Shannon Entropy | 3 | entropy_500m, entropy_1000m, entropy_2000m |
| POI Counts | 32 | competitors_count_1000m, mall_count_500m, ... |
| POI Densities | 32 | competitors_density_1000m, office_density_2000m, ... |
| Indonesia Distance | 4 | nearest_mosque_m, nearest_pasar_m, nearest_gas_station_m, ... |
| Competition | 3 | nearest_competitor_m, avg_competitor_dist_2km, cannibalization_risk_500m |
| Demographics | 3 | income_district_m, density_district, working_age_district |
| Accessibility | 3 | dist_city_center_km, transport_density_1km, urban_centrality |
| Interactions | 6 | income_pop_interaction, demand_supply_ratio, mosque_residential, ... |
| Indonesia Advanced | 4 | friday_prayer_impact, pasar_proximity_score, gas_proximity_score, market_saturation_1km |
| Temporal | 5 | ramadan_evening_multiplier, weekend_mall_multiplier, gajian_multiplier, ... |

**Total**: ~95 features + 3 labels (event_observed, survival_days, categorical_label)

---

## Expected Results

### Based on Research:

**Baseline** (demographics only):
- C-index: 0.65-0.67

**Thematic** (progressive testing):
- C-index: 0.75-0.80

**Complete Features** (this workflow):
- **Target C-index: 0.85-0.90**
- With all research-based features
- Optimized feature selection

### Likely Top Features:

Based on Phase 4 standalone and research:

1. `nearest_gas_station_m` (80% importance in Phase 4)
2. `entropy_1000m` (70% importance in simple model)
3. `transport_density_1km` (67% in Phase 3)
4. `density_district` (45% in Phase 1)
5. `working_age_district` (48% in Phase 1)

---

## Troubleshooting

### Stage 1 Errors:

**"File not found"**:
- Check dataset path
- Kaggle: Ensure dataset added to notebook
- Local: Verify `data/jakarta_clean_categorized.csv` exists

**Memory error**:
- Reduce `BUFFER_SIZES` (use only [1000, 2000])
- Skip largest buffer (5000m)
- Reduce `POI_TYPES_TO_EXTRACT` (fewer categories)

**Slow execution**:
- Check spatial indexing (STRtree should be used)
- Verify `tqdm` progress bars show
- Ensure not re-building trees unnecessarily

### Stage 2 Errors:

**"Features file not found"**:
- **Run Stage 1 first!**
- Verify `jakarta_restaurant_features_complete.csv` created
- Check file path matches (Kaggle vs Local)

**Low C-index**:
- Verify features extracted correctly (no NaN)
- Check train/test split (stratified by event_observed)
- Increase `n_estimators` (200 → 300+)

**Memory error**:
- Reduce `RSF_CONFIG['n_estimators']` (200 → 100)
- Skip some Top-K experiments (fewer K values)
- Skip progressive experiments (comment out cells 19-20)

---

## Key Differences from Other Notebooks

| Notebook | Features | Stages | C-index | Runtime |
|----------|----------|--------|---------|---------|
| `kaggle_simple_2features.ipynb` | 2 | 1 | 0.68-0.72 | ~10 min |
| `kaggle_survival_prediction_thematic.ipynb` | 15-20 | 1 | 0.75-0.80 | ~30-40 min |
| **This Workflow** | **50-60** | **2** | **0.85-0.90** | **~70-90 min** |

**Advantages**:
- ✅ Most comprehensive features (based on research)
- ✅ Reusable feature extraction
- ✅ Multiple optimization strategies
- ✅ Production-ready workflow
- ✅ Clear separation of concerns

---

## Production Deployment

After running both stages:

1. **Identify best model**:
   - Check `experiment_summary.csv`
   - Note best configuration (Top-K, All Features, etc.)
   - Note final C-index

2. **Select features**:
   - If Top-K wins: Use top N features from `feature_importance_all.csv`
   - If All Features wins: Use complete feature set

3. **Create prediction pipeline**:
   ```python
   # 1. Load new POI data
   # 2. Run feature extraction (Stage 1 code)
   # 3. Load trained model
   # 4. Predict survival probability
   # 5. Output risk scores
   ```

4. **Monitoring**:
   - Track C-index on new data monthly
   - Retrain quarterly with updated POI data
   - Monitor feature importance drift

---

## Next Steps After This Workflow

1. **Hyperparameter Tuning**:
   - Use GridSearchCV on best feature set
   - Optimize tree depth, min_samples, etc.

2. **Ensemble Methods**:
   - Combine RSF + GBS predictions
   - Weight by validation performance

3. **Feature Engineering**:
   - Create polynomial features from top predictors
   - Test additional interactions
   - Add temporal features if date data available

4. **Cross-Validation**:
   - 5-fold CV to verify stability
   - Check performance across districts

5. **Explainability**:
   - SHAP values for individual predictions
   - Partial dependence plots
   - Business rule extraction

---

## Summary

This two-stage workflow provides:

✅ **Complete feature coverage** (all research-based features)
✅ **Efficient computation** (extract once, train many times)
✅ **Clear results** (5 CSVs + 5 visualizations)
✅ **Production ready** (reusable pipeline)
✅ **High performance** (target C-index 0.85-0.90)

**Total time investment**: ~70-90 minutes
**Expected improvement**: From 0.66 baseline to 0.85+ with complete features

---

**Created**: 2025-11-18
**Status**: Ready to run
**Validated**: Sequential workflow verified, label columns aligned
