# Telecom Customer Churn Prediction

This repository contains a Jupyter Notebook that implements a customer churn prediction model using various machine learning techniques, with a primary focus on XGBoost.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Installation](#installation)
- [Notebook Structure](#notebook-structure)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [SHAP Analysis](#shap-analysis)
- [Conclusions and Recommendations](#conclusions-and-recommendations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Customer churn prediction is a critical task for businesses to proactively retain customers by identifying those who are at risk of leaving. This project focuses on building and evaluating machine learning models to predict customer churn based on their service usage patterns, account information, and demographic data.

## Dataset Overview

The dataset used in this project is provided in `Sample_Data.csv`. It contains information on customers and their interaction with the service. The key statistics are as follows:

- **Number of Records**: 1,000
- **Number of Features**: 31
- **Types of Features**:
  - **Numerical**: `age`, `tenure`, `MonthlyCharges`, `TotalCharges`, etc.
  - **Categorical**: `gender`, `Partner`, `Dependents`, `PhoneService`, etc.

## Installation

To run this notebook on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/churn-prediction.git
    cd churn-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

5. Open `Ali Amini-Churn Prediction.ipynb` to run the analysis.

## Notebook Structure

The Jupyter Notebook is organized into several key sections:

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling features.
2. **Exploratory Data Analysis (EDA)**: Visualizing data distributions, correlations, and identifying key patterns.
3. **Feature Engineering**: Creating new features that capture additional information from the existing dataset.
4. **Model Selection**: Training different machine learning models and selecting the best-performing one.
5. **Hyperparameter Tuning**: Optimizing the model's hyperparameters to improve performance.
6. **Model Evaluation**: Evaluating the model on the test set using metrics like accuracy, precision, recall, and F1 score.
7. **SHAP Analysis**: Interpreting the model using SHAP values to understand the impact of each feature.

## Data Preprocessing

Data preprocessing steps include:
- **Replacing Missing Values**: Handling missing values in `TotalCharges` and other relevant features.
- **Encoding Categorical Variables**: Converting categorical variables into a numerical format using `OneHotEncoder`.
- **Feature Scaling**: Normalizing numerical features using `MinMaxScaler`.

## Exploratory Data Analysis (EDA)

The EDA process revealed important insights:
- **Customer Demographics**: Older customers and those with month-to-month contracts are more likely to churn.
- **Service Usage**: Customers with higher monthly charges and those without additional services like `OnlineSecurity` and `TechSupport` have higher churn rates.

## Feature Engineering

New features were engineered to capture more complex relationships:
- **TenureGroup**: Categorizing customers based on their tenure.
- **TotalChargesToTenure**: Ratio of total charges to tenure.
- **NumServicesSubscribed**: Count of services a customer is subscribed to.

## Model Selection

Several machine learning models were trained and evaluated:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Linear Discriminant Analysis (LDA)**
- **XGBoost**

XGBoost was selected as the best-performing model based on its ability to handle the complexity of the dataset.

## Hyperparameter Tuning

Hyperparameters for the XGBoost model were optimized using `RandomizedSearchCV`. The best parameters found were:
- `alpha`: 0.5266
- `colsample_bylevel`: 0.9004
- `colsample_bynode`: 0.9895
- `colsample_bytree`: 0.9199
- `gamma`: 4.3350
- `lambda`: 1.8159
- `learning_rate`: 0.1755
- `max_depth`: 5
- `min_child_weight`: 8
- `n_estimators`: 492
- `scale_pos_weight`: 5.1551
- `subsample`: 0.7975

## Model Evaluation

The XGBoost model was evaluated on the test set with the following results:
- **Accuracy**: 0.5200
- **Precision**: 0.5140
- **Recall**: 0.9671
- **F1 Score**: 0.6712

## SHAP Analysis

SHAP (SHapley Additive exPlanations) analysis was conducted to interpret the model:
- **Feature Importance**: Identified key features such as `Dependents`, `Partner`, `MultipleLines`, and `Tenure` as important predictors.
- **Impact**: SHAP values provided a detailed understanding of how each feature contributed to the predictions.

## Conclusions and Recommendations

- **High Recall, Low Precision**: The model effectively identifies churners but has a high rate of false positives.
- **Actionable Insights**: Use the identified key features to target at-risk customers with personalized retention strategies.
- **Business Recommendations**: Focus on improving service offerings, particularly for high-risk customers such as those on month-to-month contracts or those with high monthly charges.

## Future Work

For future iterations of this project, consider:
- **Additional Features**: Incorporate customer satisfaction scores, competitor data, and social media sentiment.
- **Advanced Modeling**: Explore deep learning models and ensemble methods.
- **Real-Time Prediction**: Implement real-time churn prediction capabilities in customer support systems.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting issues or pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
