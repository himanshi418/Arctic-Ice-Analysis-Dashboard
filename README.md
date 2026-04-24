# ❄️ Arctic Ice Data Dashboard

An interactive **Streamlit dashboard** for analyzing Arctic Ice data with **visualizations and machine learning predictions**.

---

## 🚀 Features

- 📊 Interactive dashboard using Streamlit  
- 📈 10+ advanced visualizations  
- 🎯 Feature engineering (Date, Season, Day of Year)  
- 🤖 Machine Learning model using Random Forest  
- 📉 Ice Extent prediction system  
- 🔍 Filters (Year, Month, Season)  
- 📥 Download processed dataset  

---

## 📊 Visualizations Included

- Year vs Ice Extent (Trend)
- Month vs Ice Extent
- Distribution Histogram
- Boxplot of Ice Extent
- Correlation Heatmap
- Seasonal Doughnut Chart
- Grouped Bar Chart (Ice Concentration & Extent)
- Count Plot of Seasons
- Violin Plot (Season vs Ice Extent)
- Sea Temperature Trend

---

## 🤖 Machine Learning

Model used: Random Forest Regressor (Machine Learning)

- R² Score: ~0.99  
- Cross-validation R²: ~0.98  
- MAE: ~0.08  

The model predicts **Arctic Ice Extent** using:
- Year  
- Month  
- Day of Year  

---

## 📁 Project Structure
Arctic Dashboard
│
├── app.py
├── Arctic_Ice_Data.csv
└── requirements.txt
