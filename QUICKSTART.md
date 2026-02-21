# 🚀 Quick Start Guide
## Transboundary Air Pollution Intelligence System

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- At least 4GB RAM
- 2GB free disk space

---

## Step-by-Step Setup

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Verify Dataset
Ensure your `Dataset/` folder contains:
- 3 pollutant files: `co.csv`, `no2.csv`, `pm10.csv`
- 14 country files: `bangladesh.csv`, `germany.csv`, etc.

Each file should have columns:
- `date` (format: YYYY-MM-DD)
- `country` (country name)
- Pollutant columns: `CO`, `NO2`, `PM10`
- Location columns: `latitude`, `longitude`

### 3️⃣ Run the Pipeline

**Option A: Run All Notebooks Sequentially**
```bash
jupyter notebook
```
Then execute in order:
1. `01_data_integration_preprocessing.ipynb` → Creates master dataset
2. `02_feature_engineering.ipynb` → Generates 100+ features
3. `03_exploratory_data_analysis.ipynb` → Creates visualizations
4. `04_ml_modeling.ipynb` → Trains 15 ML models
5. `05_explainable_ai_shap.ipynb` → SHAP analysis
6. `06_spatial_analysis.ipynb` → Spatial statistics
7. `07_visualization_reporting.ipynb` → Final report

**Option B: Use Utility Functions in Your Own Script**
```python
from utils import *
import pandas as pd

# Load data
df = load_and_standardize_csv('Dataset/co.csv')

# Add features
df = add_temporal_features(df)
df = create_lag_features(df, columns=['CO'], lags=[1, 7, 30])

# Handle missing values
df = handle_missing_values(df, method='interpolate')
```

---

## 📊 Expected Runtime

| Notebook | Estimated Time | Output Size |
|----------|---------------|-------------|
| 01 - Data Integration | 2-5 min | ~50MB |
| 02 - Feature Engineering | 5-10 min | ~200MB |
| 03 - EDA | 3-5 min | ~20MB (plots) |
| 04 - ML Modeling | 10-20 min | ~100MB |
| 05 - SHAP Analysis | 5-10 min | ~50MB |
| 06 - Spatial Analysis | 3-5 min | ~20MB |
| 07 - Visualization | 5-8 min | ~30MB |

**Total:** ~45-75 minutes for full pipeline

---

## 🎯 Key Outputs

### Processed Data:
- `processed_data/master_pollution_data.pkl` - Merged dataset
- `processed_data/features_engineered.pkl` - Full feature set
- `processed_data/distance_matrix.csv` - Country distances
- `processed_data/adjacency_matrix.csv` - Neighbor relationships

### Models:
- `model_outputs/best_model_CO.pkl` - Trained CO model
- `model_outputs/best_model_NO2.pkl` - Trained NO2 model
- `model_outputs/best_model_PM10.pkl` - Trained PM10 model
- `model_outputs/model_performance.csv` - Performance metrics

### Analysis:
- `shap_outputs/neighbor_vs_self_influence.csv` - Influence quantification
- `spatial_outputs/influence_matrix_*.csv` - Flow matrices
- `spatial_outputs/gis_country_data.csv` - GIS-ready data

### Visualizations:
- `final_report/executive_dashboard.html` - Interactive dashboard
- `final_report/interactive_map_*.html` - Geographic visualizations
- `final_report/FINAL_REPORT.txt` - Comprehensive text report

---

## 🔍 Troubleshooting

### Issue: Out of Memory Error
**Solution:** Reduce sample size in notebooks or increase RAM

### Issue: Missing Data Warning
**Solution:** Check CSV column names match expected format (date, country, CO, NO2, PM10)

### Issue: SHAP Takes Too Long
**Solution:** Reduce sample size in `05_explainable_ai_shap.ipynb` (line with `shap_sample`)

### Issue: Model Performance is Poor
**Solution:** 
- Verify data quality in EDA notebook
- Check for sufficient temporal coverage (need >1 year of data)
- Ensure neighbor relationships are correctly defined

---

## 📈 Interpreting Results

### Model Performance (Notebook 04):
- **R² > 0.80** = Excellent
- **R² 0.60-0.80** = Good
- **R² < 0.60** = Needs improvement

### Neighbor Influence (Notebook 05):
- **>30%** = Strong transboundary effect
- **15-30%** = Moderate transboundary effect
- **<15%** = Weak transboundary effect

### Moran's I (Notebook 06):
- **>0.3** = Strong spatial clustering
- **0.1-0.3** = Moderate clustering
- **<0.1** = Weak/no clustering

---

## 🌐 Using with GIS Software

### ArcGIS Pro:
1. Open ArcGIS Pro
2. Add Data → `spatial_outputs/gis_country_data.csv`
3. Display XY Data (longitude, latitude)
4. Join with country shapefiles
5. Create graduated symbols based on influence scores
6. Use Flow Mapper for influence matrices

### QGIS:
1. Layer → Add Layer → Add Delimited Text Layer
2. Select `gis_country_data.csv`
3. X field: longitude, Y field: latitude
4. CRS: EPSG:4326 (WGS 84)
5. Style based on pollution or influence columns

---

## 💡 Tips for Best Results

1. **Data Quality:**
   - Ensure consistent date formats
   - Fill major gaps before running pipeline
   - Standardize country names

2. **Feature Engineering:**
   - Adjust lag periods based on your domain knowledge
   - Modify neighbor distance threshold (default: 3000km)

3. **Model Selection:**
   - XGBoost and LightGBM generally perform best
   - Try hyperparameter tuning for better results

4. **Interpretation:**
   - Focus on consistent patterns across pollutants
   - Validate findings with domain experts
   - Use SHAP for explaining specific predictions

---

## 📞 Need Help?

- Review notebook outputs and error messages
- Check `utils.py` for function documentation
- Ensure all dependencies are correctly installed
- Verify dataset structure matches expected format

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset folder contains 17 CSV files
- [ ] Ran notebooks 01-07 in sequence
- [ ] All output folders created with results
- [ ] Final report generated
- [ ] GIS data exported successfully

---

**You're Ready to Go!** 🎉

Start with notebook 01 and work through the pipeline. Each notebook builds on the previous one, so run them in order for best results.
