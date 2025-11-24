# Sampling Differences Analysis
## Phase 4 Standalone vs All-in-One Optimized

**Date**: 2025-11-18
**Issue**: Massive feature importance discrepancy between two notebook versions

---

## The Problem

### Phase 4 Standalone Results:
- **Sample size**: 72,082 mature restaurants
- **Top feature**: `nearest_spbu_m` (80.11% importance)
- **Pasar importance**: `pasar_count_1000m` (0.12% - nearly ZERO!)

### All-in-One Optimized (from Findings Report):
- **Sample size**: 50,457 mature restaurants
- **Top feature**: `pasar_count_1000m` (89.51% importance)
- **SPBU importance**: Not in top 15

### Discrepancy:
- **Sample size difference**: 72,082 vs 50,457 = **21,625 POIs difference (30%!)**
- **Feature importance**: Complete reversal - SPBU vs Pasar dominant
- **This is HUGE!**

---

## Root Cause Analysis

### 1. Same Data Source? ✅
Both use: `jakarta_clean_categorized.csv`

### 2. Same Survival Labeling Logic? ✅
Both use identical code:
```python
REFERENCE_DATE = pd.Timestamp('2024-01-01')
OBSERVATION_WINDOW_DAYS = 365 * 3

categorical_label = 2  # too new
categorical_label = 0  # failure (closed within observation)
categorical_label = 1  # success (survived observation)

df_mature = gdf_target[gdf_target['categorical_label'] != 2]
```

### 3. Different Sample Sizes - WHY?

**Hypothesis 1: Dataset Version Difference**
- Standalone might use older/newer version of jakarta_clean_categorized.csv
- **Action**: Check file modification dates

**Hypothesis 2: POI Type Filtering Difference**
- Both filter `poi_type == 'restaurant'`
- But what if total restaurant count differs?
- **Action**: Check gdf_target length before filtering

**Hypothesis 3: Date Parsing Issues**
- Maybe date_created/date_closed parsing differs
- **Action**: Check how many POIs have valid dates

**Hypothesis 4: Actually Different Datasets!**
- Standalone: Full Jakarta dataset
- All-in-one: Maybe loaded from intermediate CSV?
- **Action**: Check data loading cells

---

## Feature Count Differences

### Standalone Phase 4: 12 Features
1. mosque_count_500m
2. mosque_count_1000m
3. nearest_mosque_m
4. pasar_count_1000m
5. nearest_pasar_m
6. convenience_count_1000m
7. spbu_count_2000m
8. **nearest_spbu_m** ← DOMINANT
9. street_food_density
10. residential_compound_1000m
11. friday_traffic_index
12. local_market_strength

### All-in-One Phase 4: 4 Features (IF IT EXISTED!)
Based on report expectations:
1. mosque_count_1000m
2. **pasar_count_1000m** ← SHOULD BE DOMINANT
3. convenience_count_1000m
4. gas_station_count_1000m

**Key Difference**:
- Standalone has **DISTANCE features** (nearest_spbu_m, nearest_pasar_m)
- All-in-one has **COUNT features** only
- Distance features are continuous (0-10000m)
- Count features are discrete (0, 1, 2, ...)

---

## Impact of Feature Type on Importance

### Theory: Distance vs Count Features

**Distance Features** (nearest_X_m):
- **High cardinality**: Nearly unique value for each POI
- **Continuous**: More splits possible in tree-based models
- **Tree-friendly**: Easy to find optimal threshold
- **Result**: Often dominate feature importance

**Count Features** (X_count_1000m):
- **Low cardinality**: Many POIs have same count (0, 1, 2, 3, ...)
- **Discrete**: Limited split options
- **Less tree-friendly**: Harder to differentiate
- **Result**: Lower feature importance

### This Explains the Discrepancy!

**In Standalone Phase 4**:
- Has `nearest_spbu_m` (distance) → 80% importance
- Has `pasar_count_1000m` (count) → 0.12% importance
- Distance feature overwhelms count features!

**In All-in-One Phase 4** (expected):
- Only has count features → no distance to compete
- `pasar_count_1000m` wins by default among counts

---

## Sample Size Impact

### Question: Does sample size affect feature importance ranking?

**Theory**: YES, dramatically!

**Scenario 1: Large Sample (72K POIs)**
- More variance in features
- Distance features capture fine-grained patterns
- Model can learn complex thresholds
- **Result**: Distance features dominate

**Scenario 2: Smaller Sample (50K POIs)**
- Less variance
- Count features might be more stable
- Simpler patterns emerge
- **Result**: Count features competitive

### Statistical Effect:
- **Larger N** → More power to detect weak signals → Distance features win
- **Smaller N** → Only strong signals detected → Count features sufficient

---

## Geographic Sampling Bias

### Hypothesis: What if samples are geographically different?

**72K sample might include**:
- Suburban areas with sparse SPBU
- Areas where SPBU proximity matters for car traffic
- Lower pasar density areas

**50K sample might include**:
- Urban core only (denser)
- Higher pasar concentration
- Less car-dependent

**This would explain**:
- SPBU matters more in car-dependent suburbs (72K sample)
- Pasar matters more in urban core (50K sample)

---

## Critical Questions to Resolve

### 1. Where does the 50,457 number come from?
- Findings report claims this
- But all-in-one notebook has NO Phase 4!
- **Is the report based on standalone results that were misreported?**

### 2. What dataset was actually used for "pasar 89%" finding?
- Check outputs folder for intermediate CSVs
- Check if there's a hidden Phase 4 run

### 3. Is findings report VALID or INVALID?
- If based on 72K sample → INVALID (SPBU should dominate)
- If based on 50K sample → Need to verify where data came from

---

## Recommendation

### Immediate Actions:

1. **Check outputs folder** for any Phase 4 CSV:
   ```
   jakarta_restaurant_phase4*.csv
   phase4_feature_importance.csv
   ```

2. **Run standalone Phase 4 again** and record:
   - Total restaurants in dataset
   - Mature POI count
   - Feature importance ranking
   - Save to new file with timestamp

3. **Add Phase 4 to all-in-one** properly:
   - Use same 4 features (count only, no distance)
   - Use same buffer (1000m for pasar, mosque, convenience, gas)
   - Record sample size
   - Compare results

4. **Update findings report** with:
   - Correct sample sizes
   - Correct feature importance
   - Explanation of discrepancies
   - Warning about distance vs count features

---

## Preliminary Conclusion

**YES, this is a sampling problem, BUT it's compounded by:**

1. ✅ **Feature type difference** (distance vs count) - MAJOR IMPACT
2. ✅ **Sample size difference** (72K vs 50K) - MODERATE IMPACT
3. ⚠️ **Possible geographic bias** - UNKNOWN IMPACT
4. ❌ **Missing Phase 4 in all-in-one** - REPORT IS QUESTIONABLE

**The findings report claiming "pasar 89%" is HIGHLY SUSPICIOUS** because:
- All-in-one notebook doesn't have Phase 4
- Standalone shows SPBU 80%, pasar 0.12%
- Numbers don't match anywhere

**Action Required**: Verify the source of "pasar 89%" claim before updating report!

---

*Analysis generated: 2025-11-18*
