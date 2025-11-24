# 📦 Git Commit & Push Instructions

Panduan lengkap untuk commit pertama kali dan push ke GitHub.

---

## ✅ Files yang HARUS di-push

### 1. Source Code
```
scripts/
├── feature_extraction/
│   ├── extract_features.py
│   ├── extract_features_complete_optimized.py
│   └── ...
└── ...
```

### 2. Notebooks
```
notebooks/
├── kaggle_gbs_proven_features.ipynb           # [NEW] GBS experiments
├── kaggle_xgboost_progressive_training.ipynb  # XGBoost experiments
└── ...
```

### 3. Documentation
```
docs/
├── 00_START_HERE.md
├── RANGKUMAN_TRAINING_DAN_RENCANA.md
├── FINDINGS_CORRECTION_Phase4.md
├── ANALYSIS_Sampling_Differences.md
└── ...
```

### 4. Configuration
```
config/
├── *.yaml
├── *.json
└── ...
```

### 5. Project Files
```
README.md
STRUCTURE.md
requirements.txt
.gitignore
GIT_INSTRUCTIONS.md  # This file
```

---

## ❌ Files yang TIDAK BOLEH di-push

### 1. Data Files (2.5GB!)
```
data/
├── raw/                    # Raw OSM data
└── processed/             # Processed features
    └── features/
        └── jakarta_restaurant_features_complete.csv  # 150MB
```
**Reason**: Terlalu besar untuk GitHub (max 100MB per file)

### 2. Outputs (3.6GB!)
```
outputs/
├── kaggle_clean_data/
├── features/
└── ...
```
**Reason**: Generated files, bisa di-reproduce

### 3. Models
```
models/
├── *.pkl        # Trained models (bisa puluhan GB)
├── *.joblib
└── ...
```
**Reason**: Terlalu besar, bisa di-retrain

### 4. Environment & Cache
```
venv/           # Virtual environment
cache/          # Cache files (20MB)
logs/           # Log files
__pycache__/    # Python bytecode
*.pyc
```
**Reason**: Environment-specific, tidak perlu di-share

### 5. Temporary Files
```
nul
*.tmp
*.log
*.swp
```

### 6. IDE Files
```
.vscode/
.idea/
*.code-workspace
```

---

## 📋 Step-by-Step Instructions

### Step 1: Verify .gitignore

Pastikan `.gitignore` sudah ada dan correct:

```bash
cd D:\Script\Project\POI
cat .gitignore
```

Jika belum ada atau perlu update, file `.gitignore` sudah saya buat.

---

### Step 2: Initialize Git (if not done)

```bash
cd D:\Script\Project\POI
git init
```

Output:
```
Initialized empty Git repository in D:/Script/Project/POI/.git/
```

---

### Step 3: Check Status

Lihat files mana yang akan di-commit:

```bash
git status
```

**Expected output**:
- ✅ Banyak files di `scripts/`, `docs/`, `notebooks/`
- ❌ TIDAK ADA files di `data/`, `outputs/`, `venv/`

**Jika ada files besar yang muncul**:
```bash
# Check file sizes
git ls-files -z | xargs -0 du -h | sort -rh | head -20

# If ada file >50MB, add to .gitignore:
echo "path/to/large/file" >> .gitignore
git rm --cached path/to/large/file
```

---

### Step 4: Add Files

Add semua files yang sudah difilter oleh `.gitignore`:

```bash
git add .
```

Verify apa yang akan di-commit:

```bash
git status
```

**Review carefully**:
- ✅ Should see: scripts/, docs/, notebooks/, config/, README.md, etc.
- ❌ Should NOT see: data/, outputs/, venv/, cache/, logs/

---

### Step 5: First Commit

Commit dengan message yang descriptive:

```bash
git commit -m "Initial commit: Restaurant survival analysis project

Features:
- Feature extraction pipeline with 130 features
- Survival analysis notebooks (RSF, GBS, XGBoost)
- Comprehensive documentation and findings
- Project structure and configuration files

Models tested:
- Random Survival Forest: C-index 0.7599 (baseline)
- Gradient Boosting Survival: Testing in progress
- XGBoost survival:cox: C-index 0.41 (not suitable)

Dataset:
- 72,082 mature restaurants from Jakarta
- 130 extracted features
- 5.5% failure rate (imbalanced)

Next steps:
- Run GBS experiments on Kaggle
- Optimize hyperparameters
- Target: C-index 0.85-0.90"
```

---

### Step 6: Create GitHub Repository

1. Pergi ke GitHub: https://github.com/new
2. **Repository name**: `jakarta-restaurant-survival`
3. **Description**: Restaurant survival prediction using spatial features and survival analysis
4. **Visibility**: Public or Private (pilih sesuai kebutuhan)
5. **DO NOT** initialize with README, .gitignore, or license (sudah ada)
6. Click **Create repository**

---

### Step 7: Add Remote & Push

Copy URL dari GitHub (SSH or HTTPS):

**HTTPS** (easier):
```bash
git remote add origin https://github.com/YOUR-USERNAME/jakarta-restaurant-survival.git
```

**SSH** (recommended if setup):
```bash
git remote add origin git@github.com:YOUR-USERNAME/jakarta-restaurant-survival.git
```

Verify:
```bash
git remote -v
```

Set main branch dan push:
```bash
git branch -M main
git push -u origin main
```

**If push fails due to size**:
```bash
# Check largest files
git ls-files -z | xargs -0 du -h | sort -rh | head -20

# Remove large files from git history
git rm --cached path/to/large/file
git commit --amend -C HEAD
git push -u origin main
```

---

### Step 8: Verify on GitHub

1. Refresh GitHub repository page
2. **Should see**:
   - ✅ README.md with nice formatting
   - ✅ Folders: scripts/, docs/, notebooks/, config/
   - ✅ Files: requirements.txt, .gitignore, STRUCTURE.md
3. **Should NOT see**:
   - ❌ data/ folder
   - ❌ outputs/ folder
   - ❌ venv/ folder
   - ❌ Large .csv files

---

## 🔧 Troubleshooting

### Problem 1: File too large (>100MB)

```
remote: error: File data/xxx.csv is 150MB; this exceeds GitHub's file size limit of 100MB
```

**Solution**:
```bash
# Add to .gitignore
echo "data/xxx.csv" >> .gitignore

# Remove from git
git rm --cached data/xxx.csv

# Commit
git commit -m "Remove large file from git"

# Push again
git push
```

### Problem 2: Too many files

```
warning: adding embedded git repository: path/to/folder
```

**Solution**:
```bash
# Remove nested .git folder
rm -rf path/to/folder/.git

# Add again
git add path/to/folder
git commit --amend -C HEAD
```

### Problem 3: Authentication failed

**For HTTPS**:
```bash
# Use Personal Access Token instead of password
# Generate token at: https://github.com/settings/tokens
```

**For SSH**:
```bash
# Setup SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add to GitHub: https://github.com/settings/keys
```

---

## 🔄 Future Commits

Untuk commits berikutnya:

```bash
# Check what changed
git status
git diff

# Add changed files
git add <file>
# OR add all
git add .

# Commit with meaningful message
git commit -m "Add: [feature]"
# OR
git commit -m "Fix: [bug]"
# OR
git commit -m "Update: [what]"

# Push
git push
```

---

## 📊 Current Repository Stats

**After first push, your repo will have**:

- **~100-200 files** (source code, notebooks, docs)
- **~10-20 MB** total size (tanpa data/outputs/venv)
- **Folders**: scripts/, notebooks/, docs/, config/
- **Languages**: Python (primary), Jupyter Notebook, Markdown

**Files NOT in repo** (total ~6GB):
- data/: 2.5GB
- outputs/: 3.6GB
- venv/: ~500MB-1GB
- cache/: 20MB

---

## ✅ Checklist

Before pushing, verify:

- [ ] `.gitignore` file exists and comprehensive
- [ ] No `data/` or `outputs/` folders in `git status`
- [ ] No `venv/` or `__pycache__/` in `git status`
- [ ] No files > 50MB (check with `git ls-files -z | xargs -0 du -h | sort -rh | head -10`)
- [ ] README.md is up-to-date
- [ ] Commit message is descriptive
- [ ] Remote is set correctly
- [ ] Push successful

---

## 🎉 Done!

After successful push:

1. ✅ Your code is now backed up on GitHub
2. ✅ Others can clone and use your project
3. ✅ You can collaborate with pull requests
4. ✅ GitHub Actions can be set up for CI/CD

**Share your repo**: `https://github.com/YOUR-USERNAME/jakarta-restaurant-survival`

---

## 📝 Notes

- Data files (2.5GB) dan outputs (3.6GB) TIDAK di-push
- Untuk share data, gunakan:
  - Kaggle Datasets
  - Google Drive (dengan link di README)
  - Cloud storage (AWS S3, Azure Blob)

- Model files (.pkl) juga TIDAK di-push
- Untuk share model, gunakan:
  - Kaggle Models
  - Hugging Face Hub
  - Model registry (MLflow, Weights & Biases)

---

**Last Updated**: 2024-11-20
