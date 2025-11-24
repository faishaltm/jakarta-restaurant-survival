# How to Run Kaggle Notebooks in Background (Close Browser)

## ✅ YES - Kaggle Can Run in Background!

Unlike Google Colab, **Kaggle allows you to close your browser and continue execution** using the **"Save & Run All (Commit)"** feature.

## 🎯 Step-by-Step Guide

### Method 1: Save & Run All (Commit) - RECOMMENDED

This is the **batch/offline execution** mode that continues running after you close the browser.

#### Steps:

1. **Upload your script** to Kaggle notebook
   ```python
   # Your feature extraction script
   !python extract_features_complete_optimized.py
   ```

2. **Configure settings** (top-right):
   - **Accelerator**: GPU T4 x2 (if needed) or None (for CPU)
   - **Persistence**: Variables & Files (to save checkpoints)
   - **Internet**: On (if needed)

3. **Click "Save Version"** (top-right corner)

4. **Select "Save & Run All (Commit)"**
   - ✅ Check "Save & Run All (Commit)"
   - ❌ NOT "Quick Save" (interactive mode only)

5. **Click "Save"**

6. **Wait for confirmation**:
   - You'll see "Version X is running"
   - Notebook switches to batch execution mode

7. **Close browser tab** ✓
   - Your notebook continues running!
   - No need to keep browser open!

8. **Monitor progress**:
   - Go back to Kaggle
   - Click **"⧉ View Active Events"** (or "Your Work" → "Sessions")
   - See your running session

9. **Check output**:
   - When complete, go to your notebook
   - Click "Versions" (right panel)
   - Click on your version number
   - See output logs and download results

### Method 2: Interactive Session (NOT Recommended for Long Tasks)

**Warning**: Interactive sessions have limitations:
- ❌ Auto-stops after **1 hour of inactivity**
- ❌ Closing browser = inactivity
- ❌ Not suitable for 60+ minute tasks

Use only for testing!

## 📊 Comparison: Kaggle vs Colab

| Feature | Kaggle | Google Colab |
|---------|--------|--------------|
| Close browser & continue | ✅ Yes (Commit mode) | ❌ No |
| Max session time | 9 hours | 12 hours |
| Inactivity timeout | 1 hour (interactive) | 90 min |
| Background execution | ✅ Save & Run All | ❌ None |
| GPU quota | 30 hrs/week | Limited |
| Persistence | ✅ Built-in | ❌ Manual (Drive) |

**Verdict**: Kaggle is BETTER for long-running background tasks!

## 🔍 How to Monitor Running Sessions

### Option A: View Active Events
1. Go to Kaggle.com
2. Click **"⧉ View Active Events"** (top-right, next to notifications)
3. See all running sessions
4. Click to view logs in real-time

### Option B: Your Work → Sessions
1. Go to Kaggle.com
2. Click your profile → **"Your Work"**
3. Click **"Sessions"** tab
4. See active + completed sessions

### Option C: Notebook Versions
1. Go to your notebook
2. Right panel → **"Versions"**
3. Running versions show spinning icon
4. Completed versions show checkmark

## 💾 Enable Persistence for Checkpoints

**IMPORTANT**: Enable persistence to save your checkpoint files!

### Steps:

1. In your notebook, click **"⚙ Session Options"** (top-right)

2. Under **"Persistence"**, select:
   - **"Variables & Files"** (recommended)
   - or **"Files"** (minimum)

3. Click **"Save"**

4. **Before running** "Save & Run All (Commit)"

### What This Does:
- Saves files in `/kaggle/working/` between sessions
- Your `checkpoint_*.csv` files will persist!
- If script crashes, you can resume from checkpoint

## 📝 Complete Workflow for Your Script

### Step 1: Create Kaggle Notebook

1. Go to Kaggle.com
2. Create New Notebook
3. Upload `extract_features_complete_optimized.py`

### Step 2: Create Dataset

Upload your input data as Kaggle Dataset:
1. Go to Kaggle.com → "Create" → "New Dataset"
2. Upload `jakarta_clean_categorized.csv`
3. Name it: `jakarta-clean-categorized`
4. Click "Create"

### Step 3: Attach Dataset to Notebook

1. In your notebook, click **"+ Add Input"** (right panel)
2. Search for your dataset: `jakarta-clean-categorized`
3. Click "Add"
4. Data will be at: `/kaggle/input/jakarta-clean-categorized/`

### Step 4: Configure Notebook Settings

```python
# In first cell - Configuration
import os
os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'batch'  # Optional flag
```

Settings (top-right):
- **Persistence**: Variables & Files
- **Accelerator**: None (CPU is fine for this script)
- **Internet**: On (if downloading data)

### Step 5: Run Script

```python
# Cell 1: Install dependencies (if needed)
!pip install geopandas shapely -q

# Cell 2: Run script
!python /kaggle/working/extract_features_complete_optimized.py
```

Or directly in notebook:
```python
# Copy entire script into notebook cells
# Then click "Save & Run All (Commit)"
```

### Step 6: Save & Run All

1. Click **"Save Version"**
2. Select **"Save & Run All (Commit)"**
3. Add version notes: "Feature extraction with checkpoints"
4. Click **"Save"**
5. **Close browser** ✓

### Step 7: Check Back Later

After ~60 minutes:
1. Go to your notebook
2. Click **"Versions"** → Latest version
3. See logs and output
4. Download `jakarta_restaurant_features_complete.csv`

## 🚨 Important Limits & Tips

### Time Limits:
- **Max runtime**: 9 hours per session
- **GPU quota**: 30 hours/week
- **Idle timeout**: 1 hour (interactive only, NOT commit mode)

### Tips for Long-Running Scripts:

1. **Use checkpoints** (already in optimized script)
   - Script saves progress every section
   - Can resume if interrupted

2. **Print frequently**
   ```python
   print(f"Section 1 complete: {time.time()}")
   sys.stdout.flush()  # Force output to logs
   ```

3. **Save intermediate results**
   ```python
   df.to_csv('/kaggle/working/checkpoint.csv')
   ```

4. **Monitor memory**
   ```python
   import psutil
   print(f"RAM: {psutil.virtual_memory().percent}%")
   ```

5. **Set persistence BEFORE running**
   - Can't save what's already lost!

## 📥 Download Results

### Option A: From Notebook Output
1. Go to completed version
2. Right panel → **"Output"**
3. See generated files
4. Click **"⋮"** → **"Download"**

### Option B: Save as Dataset
1. In notebook, right panel → **"Output"**
2. Click **"Save Version as Dataset"**
3. Name it: `jakarta-restaurant-features`
4. Use in other notebooks!

### Option C: Direct Download (in script)
```python
# Add at end of script
from kaggle_secrets import UserSecretsClient
import shutil

# Copy to output for easy download
shutil.copy(
    '/kaggle/working/jakarta_restaurant_features_complete.csv',
    '/kaggle/working/output.csv'
)
```

## 🔧 Troubleshooting

### "Session stopped unexpectedly"
- Check logs for errors
- Enable persistence BEFORE running
- Reduce memory usage (lower N_JOBS)

### "Can't find checkpoint files"
- Ensure **Persistence** is enabled
- Check path: `/kaggle/working/` not `/kaggle/input/`

### "Interactive session timed out"
- Use **"Save & Run All (Commit)"** NOT interactive mode
- Commit mode doesn't have idle timeout

### "Output files not saved"
- Files must be in `/kaggle/working/`
- Enable persistence before running
- Check version output after completion

## ✅ Verification Checklist

Before closing browser:

- [ ] Dataset uploaded and attached
- [ ] Script uploaded to `/kaggle/working/`
- [ ] **Persistence** set to "Variables & Files"
- [ ] Clicked **"Save & Run All (Commit)"** (NOT Quick Save)
- [ ] See "Version X is running" message
- [ ] Verified in **"View Active Events"**

Now you can safely close the browser! ✓

## 📚 Resources

- Kaggle Docs: https://www.kaggle.com/docs/notebooks
- View Active Events: https://www.kaggle.com/notifications
- Your Sessions: https://www.kaggle.com/[username]/code

## Summary

**YES! Kaggle can run in background after closing browser!**

Key steps:
1. Use **"Save & Run All (Commit)"** NOT interactive mode
2. Enable **Persistence** for checkpoint files
3. Monitor via **"View Active Events"**
4. Download results from version **Output**

**Kaggle > Colab for long-running tasks** because of native background execution support!
