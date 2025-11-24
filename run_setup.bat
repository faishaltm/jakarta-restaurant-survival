@echo off
REM Jakarta POI MVP - Quick Setup Script
REM Runs on Windows

echo ============================================================
echo Jakarta Coffee Shop Site Selection - Quick Setup
echo ============================================================
echo.

echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 2: Checking setup status...
python setup_check.py

echo.
echo ============================================================
echo Setup Check Complete!
echo ============================================================
echo.
echo Next Steps:
echo 1. Edit .env file - set your PostgreSQL password
echo 2. Run: python src/data/init_db.py
echo 3. Run: python src/data/collect_osm.py
echo.
echo For API keys: See API_KEYS_GUIDE.md
echo.

pause
