# Kaggle Checkpoint/Resume Guide

**Problem**: Kaggle sessions timeout, forcing restart from beginning
**Solution**: Add checkpointing after each major section
**Implementation**: Save intermediate CSVs and load if they exist

---

## 🎯 Solution Overview

Instead of running all 9 sections at once, we:
1. Save intermediate results after each section
2. Check if saved file exists before running section
3. Load from checkpoint if available
4. Skip already-completed sections
5. Resume from where it stopped

---

## 📋 Checkpoint Strategy

### Checkpoints to Create (9 sections)

```
Checkpoint 0: Data loading (quick, always run)
↓ df_mature_with_labels.csv

Checkpoint 1: Shannon Entropy (slow, 15 min)
↓ df_with_entropy.csv

Checkpoint 2: POI Counts & Densities (slow, 20 min)
↓ df_with_poi_counts.csv

Checkpoint 3: Indonesia-Specific (slow, 25 min)
↓ df_with_indo_specific.csv

Checkpoint 4: Competition Metrics (fast, 5 min)
↓ df_with_competition.csv

Checkpoint 5: Demographics (fast, 1 min)
↓ df_with_demographics.csv

Checkpoint 6: Accessibility (fast, 2 min)
↓ df_with_accessibility.csv

Checkpoint 7: Interactions (fast, 3 min)
↓ df_with_interactions.csv

Checkpoint 8: Indonesia Advanced (fast, 1 min)
↓ df_with_indo_advanced.csv

Checkpoint 9: Temporal (instant, <1 min)
↓ jakarta_restaurant_features_complete.csv (FINAL)
```

---

## 🔧 Implementation Code

Add this after the Configuration cell (cell 5):

```python
# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================
print("="*80)
print("CHECKPOINT SYSTEM - Resume from Interruptions")
print("="*80)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Define checkpoint files for each section
CHECKPOINTS = {
    'labels': CHECKPOINT_DIR / '01_df_with_labels.csv',
    'entropy': CHECKPOINT_DIR / '02_df_with_entropy.csv',
    'poi_counts': CHECKPOINT_DIR / '03_df_with_poi_counts.csv',
    'indo_specific': CHECKPOINT_DIR / '04_df_with_indo_specific.csv',
    'competition': CHECKPOINT_DIR / '05_df_with_competition.csv',
    'demographics': CHECKPOINT_DIR / '06_df_with_demographics.csv',
    'accessibility': CHECKPOINT_DIR / '07_df_with_accessibility.csv',
    'interactions': CHECKPOINT_DIR / '08_df_with_interactions.csv',
    'indo_advanced': CHECKPOINT_DIR / '09_df_with_indo_advanced.csv',
    'final': OUTPUT_DIR / 'jakarta_restaurant_features_complete.csv'
}

def load_checkpoint(checkpoint_name):
    """Load a checkpoint if it exists."""
    filepath = CHECKPOINTS[checkpoint_name]
    if filepath.exists():
        print(f"✓ Loading checkpoint: {checkpoint_name}")
        return pd.read_csv(filepath, low_memory=False)
    return None

def save_checkpoint(df, checkpoint_name):
    """Save a checkpoint after completing a section."""
    filepath = CHECKPOINTS[checkpoint_name]
    df.to_csv(filepath, index=False)
    print(f"✓ Saved checkpoint: {checkpoint_name} ({len(df):,} rows)")
    gc.collect()
    return df

print(f"\n✓ Checkpoint system initialized")
print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
print(f"\n📋 Available checkpoints:")
for name, path in CHECKPOINTS.items():
    exists = "✅" if path.exists() else "⏳"
    print(f"  {exists} {name:20s} - {path.name}")
```

---

## 🚀 Modified Section Template

### BEFORE (No checkpoint)
```python
# Section code here...
df_mature['new_feature'] = calculated_values
print(f"✓ Feature extracted")
```

### AFTER (With checkpoint)
```python
# Try loading checkpoint first
loaded_df = load_checkpoint('section_name')
if loaded_df is not None:
    df_mature = loaded_df
else:
    # Section code only runs if checkpoint doesn't exist
    df_mature['new_feature'] = calculated_values
    print(f"✓ Feature extracted")

    # Save checkpoint
    df_mature = save_checkpoint(df_mature, 'section_name')
```

---

## 📝 Apply Checkpoints to Each Section

### Section 1: Shannon Entropy
**After calculating entropy, add:**
```python
print(f"\n✓ Shannon entropy extracted for {len(GRID_SIZES)} scales")
gc.collect()

# CHECKPOINT SAVE
df_mature = save_checkpoint(df_mature, 'entropy')
```

### Section 2: POI Counts & Densities
**At the very beginning, add:**
```python
# Try loading checkpoint
loaded_df = load_checkpoint('poi_counts')
if loaded_df is not None:
    df_mature = loaded_df
    print("✓ Skipping POI extraction (loaded from checkpoint)")
else:
    # Original code runs here
    print("Extracting POI counts and densities...")
    # ... all extraction code ...

    # CHECKPOINT SAVE (at end)
    df_mature = save_checkpoint(df_mature, 'poi_counts')
```

### Repeat for All Sections
Apply the same pattern to:
- Section 3: Indonesia-Specific
- Section 4: Competition
- Section 5: Demographics
- Section 6: Accessibility
- Section 7: Interactions
- Section 8: Indonesia Advanced
- Section 9: Temporal

---

## ⚡ Quick Resume Instructions

If Kaggle session times out:

1. **Click "Run from here"** at the first cell (imports)
   - Checkpoints system initializes
   - Shows which sections are complete ✅
   - Shows which sections still need to run ⏳

2. **Skip to the first incomplete section**
   - Jump to cell for incomplete section
   - Run that cell and all following

3. **Or run all from beginning**:
   - Checkpoint system automatically skips completed sections
   - Each section checks if checkpoint exists
   - Missing sections run, completed sections load instantly

---

## 💾 Storage Considerations

### Checkpoint Sizes
- Each checkpoint: ~150-200 MB (50k rows × 100 columns)
- 9 checkpoints × 150 MB = **1.35 GB total**
- Final output: 150 MB

### Kaggle Working Directory
- Default quota: 100 GB
- Checkpoints: 1.35 GB (1.35% of quota)
- Safe to keep all checkpoints ✅

### Clean Up After Done
```python
# Optional: Delete checkpoints after successful final save
import shutil
if CHECKPOINTS['final'].exists():
    print("Final output saved. Removing checkpoints to save space...")
    shutil.rmtree(CHECKPOINT_DIR)
    print(f"✓ Removed {CHECKPOINT_DIR}")
```

---

## 📊 Time Savings

### Without Checkpoints
- Session timeout at 50 min
- Must restart completely: 0-90 min all over again
- Total wasted time: 50-90 min

### With Checkpoints
- Session timeout at 50 min
- Restart and load checkpoints: 2-3 min
- Resume from checkpoint: +40 min to completion
- Total time: 50 + 3 + 40 = 93 min (normal)
- Time saved: 0-87 min (87 min in best case)

**Conclusion**: Checkpoints save time AND allow graceful recovery from interruptions.

---

## 🔍 Monitor Checkpoints

### During Execution
Watch the checkpoint messages:
```
✓ Saved checkpoint: entropy (72,082 rows)
✓ Saved checkpoint: poi_counts (72,082 rows)
✓ Saved checkpoint: indo_specific (72,082 rows)
...
```

### If Something Goes Wrong
Check checkpoint folder:
```python
# In new cell
import os
checkpoint_files = sorted(CHECKPOINT_DIR.glob('*.csv'))
for f in checkpoint_files:
    size_mb = f.stat().st_size / (1024**2)
    print(f"✓ {f.name:40s} {size_mb:6.1f} MB")
```

### Resume From Specific Point
```python
# If you want to start from a specific checkpoint
df_mature = pd.read_csv(CHECKPOINTS['entropy'], low_memory=False)
print(f"Loaded from entropy checkpoint: {len(df_mature):,} rows")
# Then run next section
```

---

## 🛡️ Safety Checks

### Verify Checkpoints Are Valid
```python
# Add validation cell after each checkpoint
for checkpoint_name in ['entropy', 'poi_counts', ...]:
    df_check = pd.read_csv(CHECKPOINTS[checkpoint_name], low_memory=False)
    assert len(df_check) == 72082, f"Checkpoint {checkpoint_name} has wrong size!"
    assert df_check.isnull().sum().sum() < 1000, f"Too many NaNs in {checkpoint_name}!"
    print(f"✓ {checkpoint_name}: {len(df_check):,} rows, columns: {len(df_check.columns)}")
```

### Final Validation
```python
# Before final save
print(f"\nFinal validation:")
print(f"  Total rows: {len(df_mature):,}")
print(f"  Total columns: {len(df_mature.columns)}")
print(f"  Missing values: {df_mature.isnull().sum().sum():,}")
print(f"  Duplicates: {df_mature.duplicated().sum()}")
assert len(df_mature) == 72082, "Row count changed!"
assert len(df_mature.columns) >= 90, "Not enough features!"
print(f"\n✅ All validations passed!")
```

---

## 📋 Checklist for Implementation

- [ ] Add CHECKPOINT SYSTEM code block after Configuration
- [ ] Modify Section 1 to save checkpoint
- [ ] Modify Section 2 to load + save checkpoint
- [ ] Modify Section 3 to load + save checkpoint
- [ ] Modify Section 4 to load + save checkpoint
- [ ] Modify Section 5 to load + save checkpoint
- [ ] Modify Section 6 to load + save checkpoint
- [ ] Modify Section 7 to load + save checkpoint
- [ ] Modify Section 8 to load + save checkpoint
- [ ] Modify Section 9 to load + save checkpoint
- [ ] Test interruption by stopping notebook halfway
- [ ] Verify it resumes correctly from checkpoint
- [ ] Add final validation cell

---

## 🎓 Example: Full Section with Checkpoint

```python
# ============================================================================
# SECTION 2: POI COUNTS & DENSITIES - WITH CHECKPOINT
# ============================================================================

# TRY LOADING CHECKPOINT FIRST
loaded_df = load_checkpoint('poi_counts')
if loaded_df is not None:
    df_mature = loaded_df
    print("✓ POI extraction skipped (loaded from checkpoint)")
    print(f"  Rows: {len(df_mature):,} | Columns: {len(df_mature.columns)}")
else:
    # ORIGINAL EXTRACTION CODE (only runs if no checkpoint)
    print("="*80)
    print("POI COUNTS & DENSITIES")
    print("="*80)
    print()

    POI_TYPES_TO_EXTRACT = {...}

    def extract_poi_features(...):
        ...

    for poi_name, poi_type in POI_TYPES_TO_EXTRACT.items():
        print(f"\nExtracting {poi_name}...")
        for buffer_m in BUFFER_SIZES:
            counts, densities = extract_poi_features(...)
            df_mature[f'{poi_name}_count_{buffer_m}m'] = counts
            df_mature[f'{poi_name}_density_{buffer_m}m'] = densities
            print(f"  {buffer_m}m: ...")

    print(f"\n✓ POI features extracted")

    # SAVE CHECKPOINT
    df_mature = save_checkpoint(df_mature, 'poi_counts')

print(f"\n✓ Section 2 complete: POI Counts & Densities")
gc.collect()
```

---

## 🚀 Running with Checkpoints

### First Run (No checkpoints)
```
✅ Loading checkpoint: entropy → Not found, extracting...
✅ Loading checkpoint: poi_counts → Not found, extracting...
...
Total time: 70-90 minutes
```

### Interrupted Then Resumed
```
✅ Loading checkpoint: entropy → Found! (2 sec)
✅ Loading checkpoint: poi_counts → Found! (2 sec)
✅ Loading checkpoint: indo_specific → Not found, extracting...
...
Total time: 40-50 minutes (saves 30-40 min)
```

---

## 📞 Troubleshooting

**Q: Checkpoint file is corrupted**
A: Delete the checkpoint file and re-run that section:
```python
import os
os.remove(CHECKPOINTS['section_name'])
# Then re-run the section cell
```

**Q: Want to re-run a section (not use checkpoint)**
A: Delete the checkpoint before running:
```python
CHECKPOINTS['section_name'].unlink(missing_ok=True)
# Now re-run the section - it will extract fresh
```

**Q: Checkpoints taking too much space**
A: Delete intermediate checkpoints, keep only latest:
```python
# Keep only the last checkpoint, delete others
keep_checkpoint = 'interactions'  # Or whatever is latest
for name, path in CHECKPOINTS.items():
    if name != keep_checkpoint and name != 'final' and path.exists():
        path.unlink()
        print(f"Deleted {name}")
```

---

## ✅ Benefits

1. ✅ **Resume from interruptions** - Don't lose progress
2. ✅ **Skip slow sections** - Load instantly from checkpoint
3. ✅ **Experiment safely** - Add new features without re-extracting
4. ✅ **Validate incrementally** - Check each section's output
5. ✅ **Clean recovery** - No corrupted partial data
6. ✅ **Time savings** - Up to 87 minutes on interruption

---

**Ready to implement checkpoints?**
→ Apply the template code above to each section
→ Test interruption/resume workflow
→ You'll thank yourself when Kaggle times out! 🙏

