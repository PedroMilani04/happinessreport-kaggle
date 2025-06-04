# World Happiness Index Forecast with XGBoost + Lag Features

This project implements a **machine learning pipeline** to forecast the **World Happiness Index for the year 2020**, based on historical data from 2015 to 2019 using **XGBoost** and **lag-based feature engineering**.

---

## 📌 Project Overview

The goal is to predict how happy each country would be in 2020 based on their historical indicators, leveraging:

1. **Data Standardization & Cleaning**  
   Each year’s dataset was cleaned to remove inconsistencies, standardize column names, and remove year-specific anomalies. Cleaned versions were saved as `YEAR_cleaned.csv`.

2. **Lag Feature Engineering**  
   Lag features were generated for each numerical column to capture temporal trends within each country (e.g., GDP_lag1, GDP_lag2).

3. **Model Training & Evaluation**  
   An **XGBoost Regressor** is trained on data from 2015–2018 and tested on 2019. Evaluation metrics include **MSE** and **R² Score**.

4. **Final Forecast for 2020**  
   The trained model is applied to the 2019 data (with year manually set to 2020) to simulate predictions for 2020.

---

## ✅ Key Features

1. **Country-Specific Lag Features**  
   Lag-1 and lag-2 values for each feature were created using `groupby('Country')`, enabling the model to learn from historical trends.

2. **Targeted Feature Set**  
   Only statistically meaningful features were selected, such as GDP, Family, Freedom, Health, etc. Unnecessary or redundant columns were removed.

3. **Robust ML Training**  
   XGBoost was chosen for its performance and ability to handle missing data and nonlinear relationships.

4. **Result Exporting**  
   Both the merged dataset and the 2020 predictions are exported as `.csv` files for further inspection or visualization.

---


## 📂 File Outputs

- `2015_cleaned.csv` to `2019_cleaned.csv`: Cleaned data per year  
- `happiness_combined.csv`: Combined dataset with lag features  
- `2020_prediction.csv`: Final 2020 happiness predictions per country

---


## 📊 Libraries Used

- `pandas`, `numpy`
- `xgboost`
- `sklearn` (metrics, model_selection, preprocessing)
- `seaborn`, `matplotlib`

---

## 📚 Conclusion

This project highlights how **lag-based feature engineering** and **gradient boosting** can be applied to time-aware forecasting of social indicators.  
By transforming static yearly data into a time series format, we enable predictive models to capture national-level trends and produce realistic future scenarios.

The framework is adaptable and can be extended with more sophisticated time-series or hybrid deep learning models in the future.

---
