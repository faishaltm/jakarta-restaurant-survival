# Hyperparameter Tuning Guide

## Overview

This guide explains how to systematically tune hyperparameters and improve model accuracy through automated experimentation and tracking.

## Quick Start

### 1. Run Full Automated Tuning

```bash
python run_hyperparameter_tuning.py
```

This will:
- Test all 5 imbalance handling strategies
- Try all hyperparameter combinations defined in `config/hyperparameter_tuning.yaml`
- Use 5-fold cross-validation for robust evaluation
- Track all experiments with metrics and configurations
- Automatically save the best model

**Expected runtime**: 2-4 hours for full grid (depending on data size and grid size)

### 2. Quick Test Mode (for debugging)

Edit `config/hyperparameter_tuning.yaml`:

```yaml
quick_test:
  enabled: true          # Enable quick test
  sample_size: 10000     # Use only 10K POIs
  n_estimators: [50]     # Fewer trees
  cross_validation_splits: 2  # Fewer CV splits
```

Then run:
```bash
python run_hyperparameter_tuning.py
```

**Expected runtime**: 10-20 minutes

---

## Configuration System

### Hyperparameter Search Space

All hyperparameters are defined in `config/hyperparameter_tuning.yaml`. You can easily modify them:

#### Example: Adjusting Random Survival Forest

```yaml
random_survival_forest:
  n_estimators:
    search_type: "grid"
    values: [50, 100, 200, 300]  # Add or remove values

  max_depth:
    search_type: "grid"
    values: [null, 10, 20, 30]  # null = unlimited

  min_samples_split:
    search_type: "grid"
    values: [5, 10, 20, 50]  # Minimum samples to split node
```

#### Example: Adjusting Gradient Boosting

```yaml
gradient_boosting_survival:
  learning_rate:
    search_type: "grid"
    values: [0.01, 0.05, 0.1, 0.2]  # Lower = more regularization

  max_depth:
    search_type: "grid"
    values: [2, 3, 5, 7]  # Shallower = less overfitting
```

#### Example: Adjusting Imbalance Handling

```yaml
sample_weighting:
  event_weight_ratio:
    search_type: "grid"
    values: [10, 20, 50, 100]  # Weight for minority class

undersampling:
  censored_to_event_ratio:
    search_type: "grid"
    values: [0.05, 0.1, 0.2, 0.3]  # 20:1, 10:1, 5:1, 3:1
```

---

## Feature Engineering Tuning

### Test Different Buffer Distances

```yaml
feature_engineering:
  buffer_distances:
    search_type: "grid"
    options:
      - [150, 500, 1000, 2000, 5000]  # Comprehensive (default)
      - [500, 1000, 2000]              # Reduced (faster)
      - [250, 750, 1500, 3000]         # Alternative spacing
```

To test this, you'll need to re-run the Jakarta Selatan pipeline with different buffer distances in `config/pipeline_config.yaml`, then run tuning on each version.

### Feature Selection

```yaml
feature_engineering:
  feature_selection:
    enabled: true
    methods:
      - "all"                    # Use all features
      - "importance_top_50"      # Top 50 by importance
      - "importance_top_30"      # Top 30 by importance
      - "remove_low_variance"    # Remove low variance features
```

---

## Experiment Tracking

### Output Directory Structure

After running tuning, you'll get:

```
outputs/hyperparameter_tuning/
└── jakarta_selatan_survival_tuning_20250114_120000/
    ├── experiment.log                    # Detailed log for debugging
    ├── all_runs.csv                       # All runs with metrics
    ├── best_run.json                      # Best configuration
    ├── best_configurations.json           # Best config per strategy
    ├── strategy_comparison.csv            # Strategy comparison
    ├── summary.json                       # Experiment summary
    ├── run_0001_hybrid_rsf/
    │   ├── config.json                    # Run configuration
    │   ├── feature_importance.csv         # Feature importance
    │   └── model.pkl                      # Trained model
    ├── run_0002_hybrid_gbsa/
    │   └── ...
    └── ...
```

### Analyzing Results

#### 1. View All Runs

```python
import pandas as pd

# Load all runs
runs = pd.read_csv('outputs/hyperparameter_tuning/[experiment_dir]/all_runs.csv')

# Sort by primary metric
runs_sorted = runs.sort_values('metric_c_index_uno', ascending=False)

# View top 10
print(runs_sorted.head(10))
```

#### 2. Find Best Hyperparameters

```python
import json

# Load best run
with open('outputs/hyperparameter_tuning/[experiment_dir]/best_run.json') as f:
    best = json.load(f)

print("Best configuration:")
print(f"  Strategy: {best['strategy']}")
print(f"  Model: {best['model_type']}")
print(f"  Hyperparameters: {best['hyperparameters']}")
print(f"  C-index (Uno): {best['metrics']['c_index_uno']:.4f}")
```

#### 3. Compare Strategies

```python
# Load strategy comparison
strategies = pd.read_csv('outputs/hyperparameter_tuning/[experiment_dir]/strategy_comparison.csv')

print("\nStrategy comparison:")
print(strategies[['strategy', 'model_type', 'c_index_uno', 'ibs']])
```

---

## Debugging and Improvement Workflow

### Step 1: Identify Issues

```bash
# Check experiment log
cat outputs/hyperparameter_tuning/[experiment_dir]/experiment.log

# Look for errors or warnings
grep ERROR outputs/hyperparameter_tuning/[experiment_dir]/experiment.log
grep WARNING outputs/hyperparameter_tuning/[experiment_dir]/experiment.log
```

### Step 2: Analyze Feature Importance

```python
import pandas as pd

# Load best model's feature importance
fi = pd.read_csv('outputs/hyperparameter_tuning/[experiment_dir]/run_0001_hybrid_rsf/feature_importance.csv')

# Top features
print("\nTop 20 features:")
print(fi.head(20))

# Low importance features (candidates for removal)
print("\nBottom 20 features:")
print(fi.tail(20))
```

### Step 3: Refine Hyperparameter Grid

Based on results, narrow down the search space:

```yaml
# Example: If best n_estimators was 200, search around it
random_survival_forest:
  n_estimators:
    search_type: "grid"
    values: [150, 200, 250]  # Narrow range

  # If max_depth=null was best, remove limited depths
  max_depth:
    search_type: "grid"
    values: [null]  # Only unlimited
```

### Step 4: Re-run Tuning

```bash
python run_hyperparameter_tuning.py
```

### Step 5: Compare Experiments

```python
# Load multiple experiment results
exp1 = pd.read_csv('outputs/hyperparameter_tuning/exp1/all_runs.csv')
exp2 = pd.read_csv('outputs/hyperparameter_tuning/exp2/all_runs.csv')

# Compare best scores
print(f"Experiment 1 best: {exp1['metric_c_index_uno'].max():.4f}")
print(f"Experiment 2 best: {exp2['metric_c_index_uno'].max():.4f}")
```

---

## Advanced: Custom Hyperparameter Search

### Add New Hyperparameter

1. Edit `config/hyperparameter_tuning.yaml`:

```yaml
random_survival_forest:
  # ... existing parameters ...

  # Add new parameter
  min_weight_fraction_leaf:
    search_type: "grid"
    values: [0.0, 0.1, 0.2]
```

2. Modify `run_hyperparameter_tuning.py` to pass it to the model (if needed)

3. Re-run tuning

### Implement Random Search

Edit `config/hyperparameter_tuning.yaml`:

```yaml
optimization:
  method: "random_search"  # Instead of grid_search

random_search:
  n_iter: 50  # Test 50 random combinations
```

This randomly samples from the hyperparameter space instead of testing all combinations.

---

## Performance Optimization Tips

### 1. Reduce Grid Size

```yaml
# Start with fewer values
random_survival_forest:
  n_estimators:
    values: [100, 200]  # Instead of [50, 100, 200, 300]

  max_depth:
    values: [null, 20]  # Instead of [null, 10, 20, 30]
```

### 2. Use Fewer CV Splits

```yaml
cross_validation:
  n_splits: 3  # Instead of 5
```

### 3. Test One Strategy at a Time

```yaml
imbalance_strategies:
  - hybrid  # Comment out others
  # - weighted
  # - ipcw
  # - undersampled
  # - standard
```

### 4. Parallel Processing

The pipeline uses all CPU cores by default (`n_jobs: -1`). Ensure your machine has enough RAM.

---

## Metrics Explanation

### Primary Metrics (for model selection)

- **C-index (Uno IPCW)**: Most robust for imbalanced survival data (0.5=random, >0.7=good)
- **C-index (Harrell)**: Standard concordance index
- **Integrated Brier Score (IBS)**: Calibration metric (lower is better, <0.2=good)

### Secondary Metrics

- **AUC at 6mo/1yr/2yr**: Time-specific discrimination (>0.7=good)
- **CV Std**: Cross-validation standard deviation (lower = more stable)

### When to Use Each Metric

- **Ranking quality**: Use C-index (Uno IPCW)
- **Calibration**: Use IBS
- **Specific time point**: Use AUC at that time
- **Model stability**: Use CV Std

---

## Common Issues and Solutions

### Issue 1: Long Runtime

**Solution**: Enable quick test mode or reduce grid size

### Issue 2: Out of Memory

**Solutions**:
- Reduce `sample_size` in quick test
- Use fewer CV splits
- Reduce feature count
- Run one strategy at a time

### Issue 3: Poor Performance (C-index < 0.6)

**Solutions**:
- Check feature engineering (are features informative?)
- Try more aggressive imbalance handling (higher event weights)
- Add more features (spatial diversity, building density)
- Check for data quality issues

### Issue 4: Overfitting (large gap between train/test)

**Solutions**:
- Increase `min_samples_leaf`
- Decrease `max_depth`
- Lower `learning_rate` for GBSA
- Use more regularization

---

## Example: Complete Improvement Cycle

```bash
# 1. Initial run (quick test)
# Edit config: quick_test.enabled = true
python run_hyperparameter_tuning.py

# 2. Analyze results
python -c "
import pandas as pd
runs = pd.read_csv('outputs/hyperparameter_tuning/.../all_runs.csv')
print(runs.nlargest(10, 'metric_c_index_uno'))
"

# 3. Refine hyperparameters based on results
# Edit config/hyperparameter_tuning.yaml with narrower ranges

# 4. Full run with refined grid
# Edit config: quick_test.enabled = false
python run_hyperparameter_tuning.py

# 5. Deploy best model
python -c "
import pickle
import json

with open('outputs/hyperparameter_tuning/.../best_run.json') as f:
    best = json.load(f)

print('Best configuration to use in production:')
print(json.dumps(best['hyperparameters'], indent=2))
"
```

---

## Next Steps

1. Run initial tuning with default settings
2. Analyze results and identify best strategy
3. Refine hyperparameter grid around best values
4. Test different feature engineering configurations
5. Deploy best model for prediction
