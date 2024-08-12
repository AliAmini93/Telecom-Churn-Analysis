# Telecom Customer Churn Prediction

## Overview

This project focuses on predicting customer churn using various machine-learning techniques. The analysis involves data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, hyperparameter tuning, and interpretability using SHAP values.

**Author**: Ali Amini

## Dataset Overview

### Dataset Summary

- **Number of Records**: 10,000
- **Number of Features**: 31
- **Types of Features**:
  - **Numerical**: `age`, `tenure`, `MonthlyCharges`, `TotalCharges`, `DataUsage`, `VoiceCalls`, `SMSCount`, `AverageChargesPerMonth`
  - **Categorical**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

## Data Preprocessing

### Handling Missing Values

The dataset was checked for missing values, and appropriate actions were taken to handle them. Missing values in the `TotalCharges` column were imputed using the median.

### Encoding Categorical Features

Categorical features were encoded using `LabelEncoder` and `OneHotEncoder` to convert them into a format suitable for machine learning models. This step was crucial for ensuring that the categorical data could be effectively used in the models.

### Data Normalization

Numerical features were normalized using `MinMaxScaler` to ensure that they were on the same scale, which is particularly important for models like SVM and K-Nearest Neighbors.

## Exploratory Data Analysis (EDA)

### Visualizations

- **Histograms**: Distribution of numerical features.
- **Box Plots**: Comparison of numerical features across churned and non-churned customers.
- **Count Plots**: Distribution of categorical features.
- **Bar Plots**: Proportion of churn within each category of the categorical features.

### Insights

- **Age and Tenure**: Older customers and those with longer tenure showed different churn rates.
- **Service Usage**: Features like `OnlineSecurity`, `TechSupport`, and `Contract` type were significant in determining churn.
- **Payment Method**: Certain payment methods showed higher churn rates.

## Feature Engineering

Several new features were engineered to capture more complex relationships:

- **Tenure Group**: Categorized customers based on their tenure.
- **Monthly Charges Bin**: Binned monthly charges into intervals.
- **Total Charges to Tenure**: Ratio of total charges to tenure.
- **Number of Services Subscribed**: Count of services a customer is subscribed to.
- **Payment Method Encoding**: One-hot encoded different payment methods.

## Model Selection

Four different models were trained and evaluated:

1. **Support Vector Machine (Linear Kernel)**
2. **K-Nearest Neighbors (KNN)**
3. **Linear Discriminant Analysis (LDA)**
4. **XGBoost**

### Training Set Results

| Model                         | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|----------|-----------|--------|----------|
| Support Vector Machine (Linear)| 0.5757   | 0.5824    | 0.5775 | 0.5799   |
| K-Nearest Neighbors            | 0.6500   | 0.6528    | 0.6620 | 0.6573   |
| Linear Discriminant Analysis   | 0.5700   | 0.5726    | 0.6000 | 0.5860   |
| XGBoost                        | 1.0000   | 1.0000    | 1.0000 | 1.0000   |

**Conclusion**: XGBoost was selected as the best-performing model due to its ability to handle the complexity of the dataset and its superior performance metrics.

## Hyperparameter Tuning and Model Evaluation

### Hyperparameter Tuning

Hyperparameters for XGBoost were optimized using `RandomizedSearchCV`. The following hyperparameters were selected:

- **alpha**: 0.5266
- **colsample_bylevel**: 0.9004
- **colsample_bynode**: 0.9895
- **colsample_bytree**: 0.9199
- **gamma**: 4.3350
- **lambda**: 1.8159
- **learning_rate**: 0.1755
- **max_depth**: 5
- **min_child_weight**: 8
- **n_estimators**: 492
- **scale_pos_weight**: 5.1551
- **subsample**: 0.7975

### Test Set Results

The model was evaluated on the test set with the following results:

- **Accuracy**: 0.5200
- **Precision**: 0.5140
- **Recall**: 0.9671
- **F1 Score**: 0.6712

**Analysis**: The model exhibits high recall, indicating it is effective at identifying churners, but at the cost of precision, leading to many false positives.

## Feature Importance

Feature importance was evaluated using XGBoost's built-in feature importance scores.

### Top Features

- **OnlineBackup_Yes**
- **Dependents_Yes**
- **TechSupport_Yes**
- **PhoneService_Yes**
- **DeviceProtection_Yes**

**Interpretation**: Features related to customer dependencies and service support are the most influential in predicting churn.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the XGBoost model:

- **Summary Plot**: Showed the impact of each feature on model output.
- **Dependence Plot**: Illustrated the relationship between the most important feature and churn prediction.

**Conclusion**: SHAP provided insights into how specific features contributed to the likelihood of churn, enhancing the interpretability of the model.

## Conclusion and Recommendations

### Business Implications

- **Targeted Retention Strategies**: The model identifies key features that contribute significantly to customer churn, such as `OnlineBackup`, `Dependents`, and `TechSupport`. These insights can be leveraged to develop targeted retention strategies. For example, customers who are not using `TechSupport` or `OnlineBackup` might be more likely to churn, suggesting that promotions or educational campaigns focused on these services could help in reducing churn rates.

- **Service Improvements**: The analysis highlights that services such as `TechSupport` and `OnlineBackup` have a strong influence on whether a customer will churn. Investing in improving these services and ensuring they are effectively marketed to customers could lead to higher customer satisfaction and lower churn rates.

- **Customer Segmentation**: By understanding the different factors that drive churn among various customer segments (e.g., those with `Dependents`, those using `MultipleLines`), the company can develop more personalized marketing and service strategies. This segmentation can help in allocating resources more efficiently, targeting the right customers with the right offers.

- **Proactive Measures**: The high recall rate of the model suggests that it is particularly effective at identifying customers who are at risk of churning. This allows the business to take proactive measures to retain these customers, such as offering discounts, enhancing service levels, or engaging with them more personally to address any concerns.

### Future Work

- **Additional Features**: Incorporate customer satisfaction scores, competitor analysis, and social media sentiment to further enhance the model’s accuracy and the insights derived from it.
- **Advanced Models**: Explore deep learning models and ensemble techniques for potentially better performance.
- **Real-Time Predictions**: Implement the model in a real-time environment to provide actionable insights for customer retention efforts.

### Deployment Considerations

- **Monitoring**: Regularly monitor the model's performance to ensure it continues to perform well over time.
- **Updating**: Periodically retrain the model with new data to capture evolving customer behaviors.
- **Integration**: Deploy the model within customer management systems for real-time churn prediction.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting issues or pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

