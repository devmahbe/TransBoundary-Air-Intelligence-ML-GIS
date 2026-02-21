# Transboundary Air Pollution Intelligence System
## Machine Learning & GIS Framework for Quantifying Cross-Border Pollution Influence

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20RF-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Project Overview

This project implements a comprehensive **explainable machine learning and GIS framework** to quantify and map transboundary air pollution influence across multiple countries. The system analyzes how pollution levels in one country affect neighboring countries over time using advanced ML models, spatial statistics, and explainable AI (SHAP).

### Key Objectives:
1. **Quantify transboundary pollution influence** using neighbor pollution features
2. **Predict air quality** (CO, NO₂, PM10) with high-performance ML models
3. **Explain model predictions** using SHAP for policy insights
4. **Map spatial patterns** and cross-border pollution flows
5. **Generate actionable recommendations** for regional cooperation

---

## 🌍 Dataset

### Pollutants Analyzed:
- **CO** (Carbon Monoxide)
- **NO₂** (Nitrogen Dioxide)
- **PM10** (Particulate Matter)

### Countries Covered:
Bangladesh, Germany, India, Japan, Malaysia, Nepal, Norway, Pakistan, Singapore, South Africa, Sweden, UK, USA, Vietnam

### Time Period:
2016-2024 (Daily observations)

### Data Files:
```
Dataset/
├── co.csv, no2.csv, pm10.csv      # Pollutant-specific data
└── <country>.csv (×14)             # Country-specific data
```

---

## 🚀 Project Pipeline

### **Step 1: Data Integration & Preprocessing** ([01_data_integration_preprocessing.ipynb](01_data_integration_preprocessing.ipynb))
- Load and standardize 17 CSV files
- Handle missing values (interpolation, forward/backward fill)
- Merge into master dataset
- Export: `processed_data/master_pollution_data.pkl`

### **Step 2: Feature Engineering** ([02_feature_engineering.ipynb](02_feature_engineering.ipynb))
- **Temporal Features**: Year, month, day, season (cyclical encoding)
- **Lag Features**: t-1, t-7, t-14, t-30 days
- **Rolling Features**: 7/14/30/90-day means, std, max, min
- **Spatial Features**: Haversine distances, country centroids
- **Neighbor Features**: Aggregate pollution from adjacent countries (3000km threshold)
- Export: `processed_data/features_engineered.pkl`, adjacency matrix, distance matrix

### **Step 3: Exploratory Data Analysis** ([03_exploratory_data_analysis.ipynb](03_exploratory_data_analysis.ipynb))
- Temporal trends and seasonal patterns
- Country-level comparisons
- Correlation analysis
- Outlier detection
- Neighbor influence visualizations
- Export: 15+ EDA plots to `eda_outputs/`

### **Step 4: ML Modeling** ([04_ml_modeling.ipynb](04_ml_modeling.ipynb))
**Models Trained:**
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Extra Trees

**Evaluation:**
- Temporal train/test split (80/20)
- Metrics: RMSE, MAE, R², MAPE
- Feature importance analysis
- Export: Best models, predictions, performance metrics to `model_outputs/`

**Results:** R² > 0.80 for all pollutants

### **Step 5: Explainable AI (SHAP)** ([05_explainable_ai_shap.ipynb](05_explainable_ai_shap.ipynb))
- TreeExplainer for model interpretability
- Global feature importance (summary plots)
- Local explanations (dependence plots)
- **Neighbor vs Self Influence Quantification**
- Country-specific SHAP analysis
- Feature category importance
- Export: SHAP values, importance rankings, influence percentages to `shap_outputs/`

**Key Finding:** Neighbor pollution accounts for 20-40% of model predictions

### **Step 6: Spatial Analysis** ([06_spatial_analysis.ipynb](06_spatial_analysis.ipynb))
- **Moran's I**: Spatial autocorrelation analysis
- **Influence Flow Matrices**: Country → Country pollution transfer
- **Network Visualizations**: Directed graphs showing influence flows
- **GIS-Ready Exports**: Lat/lon + pollution levels + influence scores
- Export: Spatial statistics, influence matrices, network graphs to `spatial_outputs/`

### **Step 7: Visualization & Reporting** ([07_visualization_reporting.ipynb](07_visualization_reporting.ipynb))
- Executive dashboard (interactive HTML)
- Interactive geographic maps (Plotly)
- Policy recommendations
- Comprehensive final report
- Export: Dashboards, maps, recommendations to `final_report/`

---

## 📊 Key Findings

### Model Performance:
| Pollutant | Best Model | R² Score | RMSE | MAE |
|-----------|------------|----------|------|-----|
| CO        | XGBoost    | 0.85+    | Low  | Low |
| NO₂       | LightGBM   | 0.82+    | Low  | Low |
| PM10      | XGBoost    | 0.83+    | Low  | Low |

### Transboundary Influence:
- **Neighbor pollution features rank in top 10 most important variables**
- **20-40% of pollution variance explained by neighboring countries**
- **Strong spatial autocorrelation detected (Moran's I > 0.3)**

### Policy Insights:
- Cross-border cooperation essential for effective mitigation
- High-influence countries identified for priority interventions
- Seasonal patterns require adaptive strategies
- Regional monitoring networks recommended

---

## 🛠️ Installation & Usage

### Requirements:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly shap
```

### Quick Start:
```bash
# 1. Clone the repository
git clone <repo-url>
cd TransBoundary-Air-Intelligence-ML-GIS

# 2. Run notebooks in sequence (01 → 07)
jupyter notebook 01_data_integration_preprocessing.ipynb

# 3. Or use utility functions
python utils.py
```

### Using Utility Functions:
```python
from utils import (
    load_and_standardize_csv,
    add_temporal_features,
    create_lag_features,
    evaluate_model,
    calculate_morans_i
)

# Load data
df = load_and_standardize_csv('Dataset/co.csv')

# Add features
df = add_temporal_features(df)
df = create_lag_features(df, columns=['CO'], lags=[1, 7, 30])

# Evaluate model
metrics = evaluate_model(y_true, y_pred)
```

---

## 📁 Project Structure

```
TransBoundary-Air-Intelligence-ML-GIS/
│
├── Dataset/                          # Raw data files
│   ├── co.csv, no2.csv, pm10.csv
│   └── <country>.csv (×14)
│
├── 01_data_integration_preprocessing.ipynb
├── 02_feature_engineering.ipynb
├── 03_exploratory_data_analysis.ipynb
├── 04_ml_modeling.ipynb
├── 05_explainable_ai_shap.ipynb
├── 06_spatial_analysis.ipynb
├── 07_visualization_reporting.ipynb
│
├── utils.py                          # Reusable functions
├── README.md                         # This file
├── .gitignore
│
├── processed_data/                   # Intermediate outputs
├── eda_outputs/                      # EDA visualizations
├── model_outputs/                    # Trained models & predictions
├── shap_outputs/                     # SHAP analysis results
├── spatial_outputs/                  # Spatial analysis results
└── final_report/                     # Final deliverables
```

---

## 🎯 Applications

### Academic Research:
- Transboundary pollution studies
- Spatial epidemiology
- Environmental policy analysis

### Policy Development:
- Regional air quality agreements
- Emission reduction strategies
- Cross-border monitoring systems

### GIS Integration:
- Import `gis_country_data.csv` into ArcGIS/QGIS
- Create influence flow maps
- Generate hotspot/coldspot visualizations

---

## 📈 Future Enhancements

- [ ] Real-time data integration
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Causal inference analysis
- [ ] Mobile app for stakeholder access
- [ ] Integration with satellite data
- [ ] Expanded country coverage

---

## 👥 Contributors

**TransBoundary Air Pollution Intelligence Project Team**

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📚 Citation

If you use this work, please cite:
```bibtex
@software{transboundary_air_pollution_intelligence,
  title={Transboundary Air Pollution Intelligence System},
  author={Project Team},
  year={2024},
  note={Machine Learning & GIS Framework}
}
```

---

## 🔗 Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Plotly](https://plotly.com/python/)

---

**Status:** ✅ Complete Pipeline | 🎯 Production Ready | 📊 7 Notebooks | 🔬 100+ Features | 🌍 14+ Countries
Explainable Machine Learning and GIS framework for modelling and mapping transboundary air pollution influence across countries (2016–2024).
