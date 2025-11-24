# 📊 Dataset Guide - Jakarta Restaurant Features

Complete guide untuk dataset yang digunakan dalam project ini.

---

## 🎯 Quick Summary

| Property | Value |
|----------|-------|
| **Filename** | `jakarta_restaurant_features_complete.csv` |
| **Size** | 89MB |
| **Rows** | 72,082 mature restaurants |
| **Columns** | 133 (130 features + 3 metadata) |
| **Failures** | 3,934 (5.5%) |
| **Successes** | 68,148 (94.5%) |
| **Format** | CSV (comma-separated) |
| **Encoding** | UTF-8 |

---

## 📥 Where to Get the Dataset

### Option 1: Kaggle Datasets (RECOMMENDED)

**If already uploaded**:
- 🔗 [Jakarta Restaurant Features on Kaggle](https://www.kaggle.com/datasets/YOUR-USERNAME/jakarta-restaurant-features)
- Click "Download" or use Kaggle API

**If not uploaded yet**:
1. Go to: https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload: `data/processed/features/jakarta_restaurant_features_complete.csv`
4. Fill details:
   - **Title**: Jakarta Restaurant Features - Survival Analysis
   - **Description**: See below
   - **Tags**: survival-analysis, geospatial, restaurants, jakarta, openstreetmap
5. Make public

**Kaggle Description Template**:
```
# Jakarta Restaurant Features - Survival Analysis

Complete feature dataset for predicting restaurant survival in Jakarta, Indonesia.

## Dataset Info
- 72,082 mature restaurants (3+ years data)
- 130 engineered features
- 5.5% failure rate (closed restaurants)
- Source: OpenStreetMap + BPS Demographics

## Features Include
- Competition metrics (count, density, distance)
- Demographics (population, income, working age)
- Accessibility (transport, distance to center)
- Indonesia-specific (pasar, mosque, gas stations)
- Spatial diversity (Shannon entropy)
- Interaction features

## Use Cases
- Survival analysis
- Restaurant site selection
- Urban analytics
- Geospatial machine learning

## Citation
GitHub: https://github.com/YOUR-USERNAME/jakarta-restaurant-survival
```

### Option 2: Google Drive

If you prefer Google Drive:

1. Upload file to Google Drive
2. Set sharing to "Anyone with link"
3. Get shareable link
4. Update README.md with link

### Option 3: GitHub Release (Not Recommended)

File is 89MB (close to 100MB limit), but possible:

```bash
# Create release
git tag -a v1.0 -m "Initial release with dataset"
git push origin v1.0

# Upload dataset.csv to release assets manually on GitHub
```

---

## 📂 File Structure

### Metadata Columns (3)

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | Restaurant name |
| `latitude` | float | Latitude coordinate |
| `longitude` | float | Longitude coordinate |

### Survival Data (2)

| Column | Type | Description |
|--------|------|-------------|
| `event_observed` | int | 1 = closed (failure), 0 = active (censored) |
| `survival_days` | int | Days from opening to event/censoring |

### Features (130)

#### 1. Shannon Entropy (3 features)
- `entropy_500m`: Diversity at 500m radius
- `entropy_1000m`: Diversity at 1km radius
- `entropy_2000m`: Diversity at 2km radius

#### 2. Competition Metrics (26 features)
**Counts by buffer**:
- `competitors_count_500m`, `competitors_count_1000m`, `competitors_count_2000m`, `competitors_count_5000m`

**Densities**:
- `competitors_density_500m`, `competitors_density_1000m`, etc.

**Distance**:
- `nearest_competitor_m`: Distance to nearest competitor

#### 3. POI Categories by Buffer (5 categories × 4 buffers = 20 features each)

Categories:
- Mall
- Office
- Transport
- Residential
- School
- Hospital
- Bank

Pattern: `{category}_count_{buffer}m`, `{category}_density_{buffer}m`

#### 4. Indonesia-Specific (12 features)
- `mosque_count_500m`, `mosque_count_1000m`, etc.
- `nearest_mosque_m`
- `pasar_count_500m`, `pasar_count_1000m`, etc.
- `nearest_pasar_m`
- `convenience_count_1000m`
- `gas_station_count_2000m`
- `nearest_gas_station_m`

#### 5. Demographics (6 features)
- `income_district_m`: Mean income at district level
- `density_district`: Population density
- `working_age_district`: Working age population %

#### 6. Accessibility (8 features)
- `transport_count_500m`, `transport_count_1000m`, etc.
- `transport_density_1km`
- `dist_city_center_km`: Distance to Jakarta center

#### 7. Interaction Features (8 features)
- `income_pop_interaction`
- `working_age_mall_inv`
- `office_transport`
- `demand_supply_ratio`
- `mosque_residential`
- `pasar_transport`
- `cannibalization_risk_500m`
- `urban_centrality`

#### 8. Indonesia Cultural Features (7 features)
- `friday_prayer_impact`
- `pasar_proximity_score`
- `gas_proximity_score`

#### 9. Temporal Multipliers (5 features)
- `market_saturation_1km`
- `ramadan_evening_multiplier`
- `ramadan_daytime_multiplier`
- `weekend_mall_multiplier`
- `gajian_multiplier`

---

## 🔧 Usage Examples

### Load in Python

```python
import pandas as pd

# Load dataset
df = pd.read_csv('jakarta_restaurant_features_complete.csv')

print(f"Shape: {df.shape}")
print(f"Failures: {df['event_observed'].sum():,}")
print(f"Features: {df.shape[1] - 5}")  # Exclude metadata + survival

# Check for missing values
print(df.isnull().sum().sum())  # Should be 0 or minimal
```

### Prepare for Survival Analysis

```python
from sksurv.util import Surv

# Metadata columns
metadata = ['name', 'latitude', 'longitude']

# Survival columns
survival_cols = ['event_observed', 'survival_days']

# Feature columns
feature_cols = [c for c in df.columns if c not in metadata + survival_cols]

print(f"Available features: {len(feature_cols)}")

# Create survival array
y = Surv.from_arrays(
    event=df['event_observed'].astype(bool),
    time=df['survival_days']
)

X = df[feature_cols].values
```

### For Kaggle Notebooks

```python
# Direct load from Kaggle dataset
from pathlib import Path

DATA_PATH = Path('/kaggle/input/jakarta-restaurant-features')
df = pd.read_csv(DATA_PATH / 'jakarta_restaurant_features_complete.csv')
```

---

## 📊 Data Quality

### Completeness
- ✅ No missing values in survival data
- ✅ Minimal missing values in features (<0.1%)
- ✅ All restaurants geolocated

### Validity
- ✅ All coordinates within Jakarta bounds
- ✅ Survival days > 0
- ✅ Feature values in expected ranges

### Consistency
- ✅ Date_created < Date_closed (for failures)
- ✅ Survival days match date calculations
- ✅ Categorical labels consistent

---

## 🔬 Data Processing Pipeline

Original → Processed:

1. **Source**: OpenStreetMap Indonesia (158,377 POIs)
2. **Filter**: Restaurants only (77,918)
3. **Survival Labeling**:
   - Reference date: 2024-01-01
   - Observation window: 3 years
   - Result: 72,082 mature restaurants
4. **Feature Engineering**:
   - Buffer analysis (500m, 1km, 2km, 5km)
   - Spatial joins
   - Distance calculations
   - Interaction terms
5. **Quality Control**: Remove invalid/outliers
6. **Output**: 72,082 × 133 CSV file

**Processing Time**: ~60-90 minutes
**Script**: `scripts/feature_extraction/extract_features_complete_optimized.py`

---

## 📈 Statistics

### Survival Distribution

| Metric | Value |
|--------|-------|
| Mean survival | 1,247 days (3.4 years) |
| Median survival | 1,095 days (3.0 years) |
| Min survival | 1 day |
| Max survival | 5,099 days (14 years) |

### Feature Ranges

Top features (proven importance):

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| `competitors_count_5000m` | 0 | 2,500+ | ~350 | ~280 |
| `nearest_competitor_m` | 0 | 5,000+ | ~180 | ~220 |
| `transport_count_5000m` | 0 | 500+ | ~85 | ~70 |
| `density_district` | 5k | 50k | ~18k | ~12k |

---

## 🚨 Known Issues

1. **Imbalanced Dataset**:
   - Only 5.5% failures
   - Requires careful handling (undersampling, weighting, or survival models)

2. **Temporal Bias**:
   - Newer restaurants haven't had time to fail yet
   - This is expected in survival analysis (censoring)

3. **Geographic Coverage**:
   - Focused on Jakarta only
   - May not generalize to other cities

---

## 💾 Storage Recommendations

### Local Storage
- **Original location**: `data/processed/features/jakarta_restaurant_features_complete.csv`
- **Size**: 89MB
- **Backup**: Recommended (irreplaceable)

### Cloud Storage
- **Kaggle**: Public dataset (shareable)
- **Google Drive**: Private backup
- **Git LFS**: If you have paid plan

### Do NOT Store In
- ❌ GitHub repository (too large)
- ❌ Email attachments (too large)
- ❌ Temporary folders (risk of loss)

---

## 🔗 Related Files

In this repository:
- `scripts/feature_extraction/extract_features_complete_optimized.py`: Generator script
- `notebooks/kaggle_gbs_proven_features.ipynb`: Usage example
- `docs/FINDINGS_CORRECTION_Phase4.md`: Feature analysis

---

## 📝 License & Attribution

**Data Source**: OpenStreetMap contributors (ODbL License)
**Demographics**: BPS Indonesia (open data)
**Processing**: This project (MIT/your license)

**Citation**:
```
Jakarta Restaurant Survival Dataset
Processed from OpenStreetMap and BPS data
GitHub: https://github.com/YOUR-USERNAME/jakarta-restaurant-survival
Year: 2024
```

---

## ❓ FAQ

**Q: Can I regenerate this dataset?**
A: Yes, run `scripts/feature_extraction/extract_features_complete_optimized.py` (~90 min)

**Q: Why 72,082 restaurants, not 77,918?**
A: Filtered out "too new" restaurants (< 3 years of data)

**Q: Can I use this for other cities?**
A: Framework yes, but need new OSM data for that city

**Q: Is this dataset updated?**
A: Static snapshot from Nov 2024. OSM updates daily.

**Q: Can I contribute more features?**
A: Yes! Edit `extract_features_complete_optimized.py` and submit PR

---

**Last Updated**: 2024-11-20
**Dataset Version**: 1.0
