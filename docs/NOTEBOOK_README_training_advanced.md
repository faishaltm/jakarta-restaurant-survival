# Advanced Survival Training Notebook - README

**File**: `kaggle_survival_training_advanced.ipynb`
**Created**: 2025-11-18
**Purpose**: Train survival models using comprehensive pre-extracted features

---

## Overview

This notebook trains and optimizes survival models using the features extracted from `kaggle_feature_extraction_complete.ipynb`.

### Strategy:
✅ **Sequential workflow** (runs AFTER feature extraction)
✅ **Multiple experiments** (Top-K, groups, progressive)
✅ **Feature importance analysis**
✅ **Optimized for performance** (target C-index 0.85-0.90)

---

## Prerequisites

**REQUIRED**: Must run `kaggle_feature_extraction_complete.ipynb` first!

**Input file**: `jakarta_restaurant_features_complete.csv`
- Should contain ~50-60 extracted features
- Includes survival labels (survival_days, event_observed)
- Mature restaurants only (categorical_label != 2)

---

## Experiment Structure

### 6 Main Experiments:

| # | Experiment | Purpose |
|---|------------|---------|
| **1** | All Features (RSF) | Baseline with Random Survival Forest |
| **2** | All Features (GBS) | Compare with Gradient Boosting |
| **3** | Top-K Selection | Test optimal number of features (5, 10, 15, 20, 30, 40, 50, all) |
| **4** | Feature Groups | Test each theme independently |
| **5** | Progressive Addition | Add groups one by one (best to worst) |
| **6** | Final Optimized | Best configuration with 300 trees |

**Total**: 30+ models tested across experiments

---

## Feature Groups Tested

### 9 Thematic Groups:

1. **Shannon Entropy**: Multi-scale POI diversity
2. **POI Counts**: Restaurant, retail, office, etc. (multiple buffers)
3. **POI Densities**: Count per km² for each category
4. **Indonesia Specific**: Mosque, pasar, convenience, SPBU
5. **Competition**: Nearest competitor, cannibalization, density
6. **Demographics**: Income, population density, working age
7. **Accessibility**: City center distance, transport, centrality
8. **Interactions**: Income×pop, mosque×residential, etc.
9. **Temporal**: Ramadan, weekend, gajian multipliers

Each group tested independently and progressively combined.

---

## Key Outputs

### CSV Files:
1. `feature_importance_all.csv` - Ranked importance of all features
2. `top_k_feature_results.csv` - Performance with different K values
3. `feature_group_results.csv` - Individual group performance
4. `progressive_feature_results.csv` - Cumulative group addition
5. `experiment_summary.csv` - Comprehensive results table

### Visualizations:
1. `feature_importance_top20.png` - Bar chart of top features
2. `top_k_performance.png` - C-index vs number of features curve
3. `feature_group_performance.png` - Group comparison
4. `progressive_addition.png` - Progressive improvement
5. `comprehensive_summary.png` - 4-panel summary dashboard

---

## Model Configuration

### Random Survival Forest (RSF):
```python
n_estimators: 200 (baseline), 300 (final)
min_samples_split: 10
min_samples_leaf: 5
max_features: 'sqrt'
max_depth: 15
```

### Gradient Boosting Survival (GBS):
```python
n_estimators: 100
learning_rate: 0.1
max_depth: 5
subsample: 0.8
```

### Train/Test Split:
- 80% train, 20% test
- Stratified by event_observed
- Random state: 42 (reproducible)

---

## Expected Performance

### Research Benchmark:
- **Target C-index**: 0.85-0.90
- **With comprehensive features** (50+ features from literature)

### Baseline Comparison:
- Phase 1 (demographics only): 0.65-0.67
- Thematic notebook (progressive): 0.75-0.80
- **This notebook (complete features)**: 0.85-0.90 target

### Key Questions:
1. **Which features matter most?** (Top-20 ranking)
2. **How many features needed?** (Optimal K)
3. **Which theme is strongest?** (Group comparison)
4. **Does progressive help?** (Incremental value)

---

## Memory Optimization

### Strategies:
1. **Load pre-extracted features** (no re-computation)
2. **Explicit garbage collection** after each experiment
3. **Delete large objects** (trained models after evaluation)
4. **Single model at a time** (no parallel model storage)

### Expected RAM Usage:
- **Data loading**: ~2-3GB
- **Model training**: ~3-4GB per model
- **Peak usage**: ~6-8GB
- **Safe for**: Kaggle T4 (16GB RAM)

---

## How to Run

### On Kaggle:
1. Upload both notebooks (extraction + training)
2. Add data: `jakarta_clean_categorized.csv`
3. Select T4 GPU (16GB RAM)
4. **Step 1**: Run `kaggle_feature_extraction_complete.ipynb` → generates CSV
5. **Step 2**: Run `kaggle_survival_training_advanced.ipynb` → trains models
6. **Expected runtime**:
   - Extraction: ~30-40 min
   - Training: ~40-50 min
   - **Total: ~70-90 min**

### Locally:
1. Ensure data in `data/jakarta_clean_categorized.csv`
2. Install: `pip install scikit-survival geopandas`
3. Run extraction notebook first
4. Verify `jakarta_restaurant_features_complete.csv` created
5. Run training notebook
6. Output in `outputs/survival_training_advanced/`

---

## Cell Breakdown (38 cells)

### Setup (Cells 1-4):
- Installation
- Imports
- Configuration
- Path setup

### Data Preparation (Cells 5-9):
- Load pre-extracted features
- Verify survival labels
- Identify feature columns
- Handle missing values
- Train/test split

### Baseline Models (Cells 10-11):
- RSF with all features
- GBS with all features

### Feature Analysis (Cells 12-13):
- Feature importance ranking
- Top-20 visualization

### Top-K Experiments (Cells 14-15):
- Test K = 5, 10, 15, 20, 30, 40, 50, all
- Performance curve

### Feature Groups (Cells 16-18):
- Define 9 thematic groups
- Test each independently
- Group comparison chart

### Progressive Addition (Cells 19-20):
- Add groups best-to-worst
- Track cumulative improvement

### Final Model (Cells 21-22):
- Select best configuration
- Train with 300 trees
- Final C-index

### Summary & Reporting (Cells 23-26):
- Comprehensive results table
- Multi-panel visualization
- Key findings
- Recommendations

---

## Validation Checklist

Before running, verify:
- [ ] Feature extraction completed successfully
- [ ] `jakarta_restaurant_features_complete.csv` exists
- [ ] File has ~50-60 feature columns
- [ ] Survival labels present (survival_days, event_observed)
- [ ] Kaggle T4 GPU selected (or local 16GB+ RAM)
- [ ] Expected runtime: 40-50 min (budget accordingly)

---

## Troubleshooting

### If "File not found" error:
1. Verify feature extraction notebook ran successfully
2. Check output path (Kaggle: /kaggle/input, Local: data/)
3. Ensure CSV saved correctly

### If Memory Error:
1. Reduce `n_estimators` in RSF_CONFIG (200 → 100)
2. Skip some Top-K values (test fewer K)
3. Skip progressive experiments (comment out cells 19-20)

### If Slow:
1. Reduce tree count (`n_estimators`)
2. Use fewer Top-K experiments
3. Skip feature group experiments
4. Check `n_jobs=-1` using all cores

### If Low C-index:
1. Check feature extraction worked (no NaN)
2. Verify train/test split stratification
3. Try different random_state
4. Increase `n_estimators` to 300+

---

## Comparison with Other Notebooks

| Notebook | Purpose | Features | Runtime | C-index Target |
|----------|---------|----------|---------|----------------|
| `kaggle_simple_2features.ipynb` | Baseline | 2 (entropy + density) | ~10 min | 0.68-0.72 |
| `kaggle_survival_prediction_thematic.ipynb` | Progressive testing | 15-20 (thematic) | ~30-40 min | 0.75-0.80 |
| `kaggle_feature_extraction_complete.ipynb` | **Feature prep** | **50-60 (all research)** | **~30-40 min** | **N/A** |
| **`kaggle_survival_training_advanced.ipynb`** | **Optimized training** | **50-60 (pre-extracted)** | **~40-50 min** | **0.85-0.90** |

**This workflow** is unique because:
- ✅ Separates feature extraction from training (reusable)
- ✅ Tests comprehensive feature set from research
- ✅ Multiple optimization strategies (Top-K, groups, progressive)
- ✅ Production-ready final model selection

---

## Next Steps After Running

### 1. Identify Best Configuration:
- Check `experiment_summary.csv`
- Compare C-index across all experiments
- Note optimal K value and best feature group

### 2. Production Deployment:
- Use final model configuration
- Save top features list
- Create prediction pipeline

### 3. Further Optimization:
- Hyperparameter tuning (GridSearchCV)
- Ensemble methods (RSF + GBS combined)
- Cross-validation for stability

### 4. Feature Engineering:
- Create interactions from top features
- Test polynomial features
- Add temporal features if date data available

---

## Key Insights to Expect

Based on research and Phase 4 analysis:

**Likely Top 5 Features**:
1. Gas station distance (`nearest_spbu_m`) - 80% in Phase 4
2. Transport density - 67% in Phase 3
3. Shannon entropy (multi-scale) - 70% in simple model
4. Competition density - Strong in Phase 2
5. Demographics (income/density) - 45-48% in Phase 1

**Likely Best Group**: Indonesia Specific or Accessibility

**Optimal K**: Likely 15-25 features (diminishing returns after)

**Final C-index**: 0.82-0.88 (realistic with complete features)

---

## Sequential Workflow Summary

```
Step 1: kaggle_feature_extraction_complete.ipynb
        ↓
        Generates: jakarta_restaurant_features_complete.csv
        ↓
Step 2: kaggle_survival_training_advanced.ipynb
        ↓
        Outputs: Models + Analysis + Recommendations
```

**Total Time**: ~70-90 minutes
**Total Output**: Feature CSV + 5 result CSVs + 5 visualizations

---

## Contact & Issues

If errors occur:
1. Check feature extraction completed
2. Verify CSV path and format
3. Review cell execution order
4. Check memory usage
5. Verify scikit-survival version

**Generated**: 2025-11-18
**Status**: Ready to run (AFTER feature extraction)
**Validated**: Structure checked, sequential workflow verified
