#!/bin/bash
# Run this from the root of your Capstone_project directory

echo "---- Backup (recommended) ----"
mkdir -p backup_project
cp -r ./* backup_project/

echo "---- Cleaning duplicate/extra files ----"
rm -f requirements\ copy.txt
find . -maxdepth 1 -type f -name "script_*.py" -delete
find . -maxdepth 1 -type f -name "*.jpg" -delete
find . -maxdepth 1 -type f -name "README copy.md" -delete

echo "---- Organizing data files ----"
mkdir -p data
mv sample_courses_*.csv data/ 2>/dev/null
mv sample_courses_*.json data/ 2>/dev/null
mv test_courses_processed.csv data/ 2>/dev/null

echo "---- Installing dependencies ----"
pip install -r requirements.txt

echo "---- Running the Streamlit app ----"
streamlit run app.py
