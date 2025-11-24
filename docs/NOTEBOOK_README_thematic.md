# Thematic Survival Prediction Notebook - README

**File**: `kaggle_survival_prediction_thematic.ipynb`
**Created**: 2025-11-18
**Purpose**: Progressive feature testing for restaurant survival prediction

---

## Overview

This notebook tests features **one theme at a time** to understand their individual and cumulative impact on survival prediction.

### Strategy:
✅ **Linear progression** (not cross-grid)
✅ **Memory optimized** for Kaggle T4 GPU (16GB RAM)
✅ **Clear tracking** of C-index improvements
✅ **Thematic grouping** of features

---

## Experiment Structure

### 7 Progressive Experiments:

| # | Experiment | Features Added | Purpose |
|---|------------|---------------|---------|
| **1** | Demographics Only | income, density, working_age | Baseline with district data |
| **2** | + Competition (1km) | competitors, nearest_competitor, anchor POIs | Add market dynamics |
| **3** | + Accessibility | dist_city_center, transport_density | Add location quality |
| **4** | + Indonesia-Specific | mosque, pasar, convenience, gas_station (COUNT only) | Add local context |
| **5** | + Interactions | mosque×residential, demand_supply, pasar×transport | Add derived features |
| **6** | Buffer Test (2km) | Re-extract key features at 2km | Test spatial scale |
| **7** | Buffer Test (5km) | Re-extract key features at 5km | Test larger scale |

**Total**: 7 experiments, ~15-20 features tested

---

## Key Design Decisions

### 1. Why COUNT only for Indonesia-Specific (EXP4)?
- Avoid distance feature dominance (we know nearest_spbu_m gets 80%)
- Test if pasar/mosque COUNT is valuable alongside other features
- Fair comparison across feature themes

### 2. Why test buffers separately (EXP6-7)?
- Avoid combinatorial explosion (3 buffers × 20 features = 60 features!)
- Understand buffer impact independently
- Memory efficient for Kaggle

### 3. Why only 3 interactions (EXP5)?
- Most impactful interactions based on Phase 5 analysis
- `mosque_residential`: 30% importance
- `demand_supply_ratio`: 30% importance
- `pasar_transport`: NEW - test if pasar matters in interaction

---

## Expected Outcomes

### Baseline (Phase 1):
- C-index: ~0.65-0.67
- Just demographics

### Target (All Features):
- C-index: **0.75-0.80**
- Full feature set

### Key Questions:
1. **Which theme adds most value?** (Demographics vs Competition vs Indonesia)
2. **Do interactions help?** (EXP5 vs EXP4)
3. **What's optimal buffer?** (1km vs 2km vs 5km)
4. **Is pasar valuable?** (When tested as COUNT with other features)

---

## Memory Optimization

### Strategies Used:

1. **Explicit garbage collection** after major operations
   ```python
   del large_object
   gc.collect()
   ```

2. **Reuse spatial trees** (don't rebuild for each buffer)

3. **Progressive loading** (don't load all features at once)

4. **Single buffer extraction** (not 4 buffers simultaneously)

5. **Minimal intermediate storage** (compute on-the-fly)

### Expected RAM Usage:
- **Data loading**: ~2-3GB
- **Feature extraction**: ~4-6GB peak
- **Model training**: ~2-3GB
- **Total**: ~8-10GB (safe for 16GB Kaggle T4)

---

## Output Files

After running:
1. `thematic_experiments_summary.csv` - Results table
2. `experiment_progression.png` - C-index progression chart

**Columns in CSV:**
- Experiment name
- Number of features
- C-index (Harrell)
- C-index (Uno)
- Training time
- Improvement over EXP1

---

## How to Run

### On Kaggle:
1. Upload notebook
2. Add data: `jakarta_clean_categorized.csv`
3. Select T4 GPU (16GB RAM)
4. Run all cells
5. **Expected runtime**: ~30-40 minutes

### Locally:
1. Ensure data in `data/jakarta_clean_categorized.csv`
2. Install: `pip install scikit-survival geopandas`
3. Run cells sequentially
4. Output in `outputs/survival_prediction_thematic/`

---

## Cell Breakdown (33 cells)

### Setup (Cells 1-10)
- Installation
- Imports
- Configuration
- Data loading
- Survival labels
- Helper functions

### Experiments (Cells 11-25)
- **EXP1**: Demographics (1 cell)
- **EXP2**: + Competition (1 cell)
- **EXP3**: + Accessibility (1 cell)
- **EXP4**: + Indonesia (1 cell)
- **EXP5**: + Interactions (1 cell)
- **EXP6**: Buffer 2km (1 cell)
- **EXP7**: Buffer 5km (1 cell)

### Analysis (Cells 26-33)
- Summary table
- Visualization
- Key findings
- Conclusion

---

## Validation Checklist

Before running, verify:
- [ ] Data file exists and path is correct
- [ ] Kaggle T4 GPU selected (or local machine has 16GB+ RAM)
- [ ] scikit-survival installed
- [ ] Expected runtime: 30-40 min (budget accordingly)

---

## Troubleshooting

### If Memory Error:
1. Reduce `RSF_CONFIG['n_estimators']` from 100 to 50
2. Add more `gc.collect()` calls
3. Use smaller sample (add `.sample(frac=0.5)`)

### If Slow:
1. Check `n_jobs=-1` is using all cores
2. Reduce POI types in competition features
3. Skip buffer experiments (EXP6-7)

### If NaN in features:
- Check data loading (missing POI categories?)
- Verify district mappings
- Check Indonesia keyword detection

---

## Next Steps After Running

1. **Identify best experiment** (highest C-index)

2. **Optimize that config**:
   - Add more trees
   - Tune hyperparameters
   - Add more interactions

3. **Test additional features**:
   - Distance to SPBU (if not memory constrained)
   - More derived features
   - Temporal features (if available)

4. **Create final production model**:
   - Train on full dataset
   - Save model
   - Deploy for predictions

---

## Comparison with Other Notebooks

| Notebook | Purpose | Features | Runtime | C-index |
|----------|---------|----------|---------|---------|
| `kaggle_simple_2features.ipynb` | Minimal model | 2 (entropy + density) | ~10 min | 0.68-0.72 |
| `kaggle_phase4_indonesia_specific.ipynb` | Indonesia features only | 12 (distance + count) | ~20 min | Feature importance |
| `kaggle_phases_all_in_one.ipynb` | All phases (feature importance) | 22 | ~40 min | Feature ranking |
| **`kaggle_survival_prediction_thematic.ipynb`** | **Progressive testing** | **15-20 (thematic)** | **~30-40 min** | **0.75-0.80 target** |

**This notebook** is unique because:
- ✅ Actually trains survival models (not just feature importance)
- ✅ Tests features progressively (not all at once)
- ✅ Optimized for memory and runtime
- ✅ Clear comparison between themes

---

## Contact & Issues

If errors occur:
1. Check data path
2. Verify Kaggle environment
3. Review cell execution order
4. Check memory usage

**Generated**: 2025-11-18
**Status**: Ready to run
**Validated**: Structure checked, cells verified
