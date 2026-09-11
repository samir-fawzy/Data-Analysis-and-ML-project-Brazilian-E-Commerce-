# Brazilian E-Commerce — Data Analysis & Machine Learning

A data analysis and machine learning project built using the **Brazilian E-Commerce (Olist)** dataset.

The project focuses on understanding order and delivery behavior, performing feature engineering, and experimenting with machine learning models for **delivery time prediction**.

## Project Overview

The main objective of this project is to analyze e-commerce order data and build machine learning models that can predict delivery-related outcomes.

The project started as a **regression problem**, where the target was:

> **Predict the delivery time of an order in days.**

Several machine learning algorithms were evaluated, including Decision Tree, Random Forest, Gradient Boosting, Linear Regression, and XGBoost.

A secondary **classification experiment** was also explored to predict whether an order would be delivered late.

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**.

The data was transformed from multiple order-related tables into an order-level dataset.

Key information includes:

* Order purchase information
* Customer state
* Order price
* Freight cost
* Number of items
* Estimated delivery information
* Delivery time

## Data Preparation

The data preparation process included:

* Combining and aggregating order-level information
* Handling missing values
* Removing duplicate records
* Converting timestamp columns
* Creating time-based features
* Creating delivery-related features
* Detecting skewness and outliers
* Encoding categorical variables
* Scaling numerical variables when required by the algorithm
* Splitting the data into training and testing sets

## Feature Engineering

Several features were created from the original timestamps and order information, including:

* `purchase_hour`
* `purchase_day_of_week`
* `purchase_month`
* `purchase_is_weekend`
* `purchase_hour_interval`
* `estimated_delivery_window_days`

Additional order-related features were also explored, such as aggregated price, freight, and item count.

## Regression Models

The following regression algorithms were tested:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

### Best Regression Result

The best result obtained during the experiments was achieved using **XGBoost**.

| Metric |         Result |
| ------ | -------------: |
| MAE    | **4.139 days** |
| RMSE   | **7.104 days** |
| R²     |      **0.327** |

The model was further tuned using `GridSearchCV`.

Best parameters:

```text
max_depth = 7
min_child_weight = 10
n_estimators = 200
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 1.0
```

A logarithmic transformation of the target was also tested because the delivery-time distribution was highly right-skewed. While it improved MAE, it reduced RMSE and R², so the original target scale was retained for the final regression benchmark.

## Classification Experiment

A second approach was explored by converting the problem into a binary classification task:

```text
0 → Not Late
1 → Late
```

Because the dataset was highly imbalanced, evaluation focused on:

* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

The initial Decision Tree classifier achieved:

```text
ROC-AUC ≈ 0.778
```

Threshold analysis was also performed to understand the trade-off between precision and recall.

## Technologies & Libraries

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
```

## Requirements

Install the required libraries with:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## Project Structure

```text
brazilian-ecommerce/
│
├── notebooks/
│   └── analysis_and_modeling.ipynb
│
├── data/
│
├── requirements.txt
├── README.md
└── ...
```

## Evaluation

For regression, the main evaluation metrics used were:

* **MAE (Mean Absolute Error)** — average prediction error in days
* **RMSE (Root Mean Squared Error)** — gives higher weight to large errors
* **R² Score** — measures how much of the target variation is explained by the model

For classification, the evaluation focused on metrics suitable for imbalanced data rather than relying on accuracy alone.

## Key Findings

The exploratory analysis showed that delivery time is highly right-skewed, with a relatively small number of orders taking significantly longer than most orders.

The estimated delivery window was one of the strongest numerical predictors of delivery time, while customer state also showed substantial differences in average delivery behavior.

Model comparison showed that tree-based boosting models performed better than the linear regression baseline on this dataset.

## Notes

This project is primarily a **learning and experimentation project** focused on understanding the complete machine learning workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
```

The project intentionally explores multiple approaches rather than presenting a production-ready prediction system.
