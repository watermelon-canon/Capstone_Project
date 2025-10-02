#!/bin/bash

echo "Backing up current project to backup_before_cleanup/"
mkdir -p backup_before_cleanup
cp -r ./* backup_before_cleanup/

echo "Cleaning duplicates and organizing files..."
rm -f requirements\ copy.txt
rm -f README\ copy.md
rm -f *.jpg
rm -f script_*.py
mv sample_courses_*.csv data/ 2>/dev/null
mv sample_courses_*.json data/ 2>/dev/null
mv test_courses_processed.csv data/ 2>/dev/null

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Launching Streamlit application..."
streamlit run app.py
