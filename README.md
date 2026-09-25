# SIT720 8.1D – Machine Learning Mini Project

## Sydney Housing Price Prediction and Decision Support System

This repository contains the materials for **SIT720 – Machine Learning Mini Project (8.1D)**.

The project develops a machine learning system for predicting residential property sale prices in selected Sydney suburbs and demonstrates how the predictions can support property price decision-making.

---

## Project Overview

The project uses a dataset containing **120 Sydney property observations** from:

- Blacktown
- Castle Hill
- Coogee

The dataset includes property characteristics such as:

- Town/Suburb
- Property type
- Number of bedrooms
- Number of bathrooms
- Parking spaces
- Land area
- Sold date
- Sold price

The project covers data exploration, preprocessing, model development, evaluation, error analysis, and deployment considerations.

---

## Machine Learning Models

Three regression models were developed and compared:

1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Five-fold cross-validation was also used to assess model performance.

---

## Model Results

The Random Forest model produced the strongest held-out test performance among the three models used in this project.

### Held-out Test Results

| Model | MAE (AUD) | RMSE (AUD) | R² |
|---|---:|---:|---:|
| Linear Regression | 770,086.72 | 888,115.75 | 0.04 |
| Random Forest | 617,892.11 | 818,296.26 | 0.18 |
| Gradient Boosting | 788,764.04 | 1,038,454.81 | -0.32 |

The Random Forest model achieved a held-out **MAE of approximately AUD 617,892** and an **R² of 0.18**.

The results also showed substantial prediction errors for some properties, highlighting the limitations of the available dataset and features.

---

## Project Structure

```text
8.1-D/
│
├── README.md
├── SIT720_8.1D_Submission_Ready_Notebook_EXECUTED_FINAL.ipynb
├── SIT720_8.1D_Report.pdf
├── SIT720_8.1D_app.py
├── SIT720_8.1D_requirements.txt
├── SIT720_Sydney_Housing_Coogee_CastleHill_Blacktown_120_rows.csv
├── sydney_housing_rf_pipeline.joblib
│
└── figures/
    └── project figures and visualisations