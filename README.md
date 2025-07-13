#CelebalTechAssignmentWeek6
# House Price Prediction – Model Evaluation & Hyperparameter Tuning

This project evaluates and tunes multiple regression models to predict house sale prices. Using the processed data from a previous preprocessing pipeline, this project compares model performances, applies hyperparameter tuning, and visualizes the results to select the best-performing model.

---

## 📁 Project Structure
<br>
CelebalTechAssignmentWeek6 
├── data/
│ └── processed_train.csv # Cleaned dataset (features + target)
├── model_train.py # Full pipeline script
├── results/
│ ├── metrics_summary.csv # Metrics from base models
│ ├── tuned_metrics_summary.csv # Metrics from tuned models
│ └── tuning_results.csv # Best parameters for each model
├── visuals/
│ ├── residuals_<model>.png # Residual histogram
│ ├── actual_vs_predicted_<model>.png # Scatterplot of true vs predicted
│ ├── r2_score_comparison.png # Barplot: R² of all models
│ ├── rmse_comparison.png # Barplot: RMSE of all models
│ └── best_model_confusion.png # Final highlight of best model
├── house_modeling.ipynb # Jupyter notebook version with code & output
├── requirements.txt # Required libraries
├── LICENSE # MIT License
└── README.md # This file
</br>
---


## 🚀 Workflow Overview

### Step 1: Load Data
- Load `processed_train.csv`
- Split into `X` (features) and `y` (target)
- Create train-test split (80/20)

### Step 2: Define Models
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbors (KNN)  
- Linear Regression

### Step 3: Model Evaluation
- Custom evaluation function to compute:
  - R² Score  
  - RMSE  
  - MAE  
- Saves visualizations:
  - Residual distribution
  - Predicted vs actual plots

### Step 4: Hyperparameter Tuning
- GridSearchCV or RandomizedSearchCV applied to all models
- Optimal parameters stored and evaluated again

### Step 5: Results Summary
- Performance metrics for both base and tuned models saved
- Comparison plots generated
- Best model automatically highlighted and visualized

---

## 📊 Key Metrics Used

| Metric         | Description                       |
|----------------|-----------------------------------|
| R² Score       | Goodness of fit                   |
| RMSE           | Root Mean Squared Error           |
| MAE            | Mean Absolute Error               |

---

## 📈 Visual Outputs

| File                                  | Description                             |
|---------------------------------------|-----------------------------------------|
| `residuals_<model>.png`              | Histogram of prediction residuals       |
| `actual_vs_predicted_<model>.png`    | Scatterplot of predictions vs ground truth |
| `r2_score_comparison.png`            | R² scores of all tuned models           |
| `rmse_comparison.png`                | RMSE of all tuned models                |
| `best_model_confusion.png`           | Highlighted visual for top model        |

---

---

## 🔍 Model Analysis

All models were trained and evaluated using consistent data and metrics. Based on the tuned performance:

- **Top Models (by R² Score):**
  - XGBoost
  - Gradient Boosting
  - Random Forest

- **Insights:**
  - XGBoost consistently delivered the best generalization, with the highest R² score and lowest RMSE.
  - Gradient Boosting and Random Forest performed closely behind, with slightly higher error margins.
  - SVR and KNN struggled to capture complex feature interactions, resulting in underperformance.
  - Linear Regression showed high bias, indicating the need for more expressive models for this dataset.

---

## 🏆 Best Model: XGBoost

The **XGBoost Regressor** emerged as the best-performing model based on:

| Metric     | Value |
|------------|-------|
| R² Score   | *[your XGBoost R² value]*  
| RMSE       | *[your XGBoost RMSE value]*  
| MAE        | *[your XGBoost MAE value]*  

### Visualization:

| Visual                               | Description                              |
|--------------------------------------|------------------------------------------|
| `visuals/best_model_confusion.png`   | Actual vs Predicted plot for XGBoost     |

This plot shows how closely the model's predictions align with actual house prices. A well-aligned diagonal line indicates high accuracy and minimal bias.

---


