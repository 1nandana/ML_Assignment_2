# Bank Marketing Subscription Prediction

## 1. Project Overview

This project compares five supervised machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit.

The models implemented are:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 222  
**Number of instances:** 45,211  
**Input features used:** 15  
**Target variable:** `y`

The target variable indicates whether a customer subscribed to a term deposit (`yes` or `no`).

The `duration` feature was excluded because it represents call duration and would not be available before making the call.

## 3. Data Preprocessing

The following preprocessing steps were applied:

- Checked for missing values and duplicate records.
- Separated input features and target variable.
- Split the dataset into 80% training data and 20% testing data.
- Used stratified splitting to preserve class distribution.
- Applied median imputation and StandardScaler to numerical features.
- Applied most-frequent imputation and OneHotEncoder to categorical features.
- Fitted preprocessing transformations on training data only.

## 4. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8937 | 0.7512 | 0.6714 | 0.1796 | 0.2834 | 0.3100 |
| Decision Tree | 0.8306 | 0.6101 | 0.2950 | 0.3223 | 0.3080 | 0.2120 |
| KNN | 0.8876 | 0.6915 | 0.5568 | 0.1947 | 0.2885 | 0.2826 |
| Naive Bayes | 0.8466 | 0.7340 | 0.3689 | 0.4376 | 0.4003 | 0.3146 |
| Support Vector Machine | 0.8932 | 0.7013 | 0.6503 | 0.1881 | 0.2918 | 0.3105 |

## 5. Model Observations

**Logistic Regression:** Achieved the highest accuracy (0.8937), AUC (0.7512) and precision (0.6714). However, its recall was relatively low.

**Decision Tree:** Achieved an accuracy of 0.8306 and recall of 0.3223. It had the lowest AUC and MCC among the five models.

**K-Nearest Neighbors:** Achieved an accuracy of 0.8876. Its precision was 0.5568, but recall was relatively low.

**Naive Bayes:** Achieved the highest recall (0.4376), F1 Score (0.4003) and MCC (0.3146). It provided the best balance of precision and recall based on the F1 Score.

**Support Vector Machine:** Achieved an accuracy of 0.8932 and precision of 0.6503. Its recall was relatively low compared with Naive Bayes.

## 6. Best Performing Model

Naive Bayes performed best based on F1 Score (0.4003) and MCC (0.3146). It also achieved the highest recall (0.4376).

Logistic Regression achieved the highest accuracy, AUC and precision. Therefore, the choice of the best model depends on the evaluation metric and the importance of identifying customers who are likely to subscribe.

Based on F1 Score and MCC, Naive Bayes is selected as the overall preferred model for this comparison.

## 7. Repository Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── logistic_regression.joblib
├── decision_tree.joblib
├── knn.joblib
├── naive_bayes.joblib
├── support_vector_machine.joblib
└── preprocessor.joblib
```

## 8. How to Run the Application Locally

1. Clone the repository.
2. Install the required dependencies.
3. Run the Streamlit application.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a test CSV file containing the input features and target column `y`. Select a trained model to view its evaluation metrics and confusion matrix.

The application loads the saved models and preprocessing artifact; it does not retrain models on uploaded test data.

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 10. Project Links

**GitHub Repository:** https://github.com/1nandana/ML_Assignment_2
**Streamlit Application:**
