# Course Equivalency Mapping System

A production-ready Streamlit application for analysing and ranking course equivalency across higher-education institutions.

## Features
- **Semantic similarity engine** combining transformer embeddings (with CPU-friendly fallbacks), structural analysis, and learning outcome matching.
- **Course processor** that normalises catalog data, extracts metadata, and validates input files.
- **Interactive Streamlit UI** for uploading catalogs, tuning similarity weights, inspecting ranked matches, and exporting reports.
- **Visual analytics** using Plotly heatmaps and component breakdown charts to aid interpretation.
- **Sample dataset** of 30+ diverse courses covering computing, mathematics, business, and humanities disciplines.

## Getting started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Execute the test suite**
   ```bash
   pytest
   ```

## Project structure
```
app.py
src/
  config/
  models/
  ui/
  utils/
data/
docs/
tests/
```

See `docs/overview.md` for a deeper architectural explanation.

## Testing
Unit tests cover the core data loading, processing, and similarity scoring workflows. Add further tests in the `tests/` directory to extend coverage as new features are implemented.

## Repository sync status

This workspace currently tracks only the local `work` branch. No Git remotes are configured, so the latest changes have **not** been pushed to any external repository. To publish the commits, add the appropriate remote and push manually:

```bash
git remote add origin <your-remote-url>
git push -u origin work
```
