# Diabetes Prediction Using Machine Learning

## **Project Overview**
This project predicts the presence of diabetes in patients using two machine learning models:  

- **Decision Tree Classifier** – interpretable, rule-based model  
- **K-Nearest Neighbors (KNN) Classifier** – distance-based model requiring feature scaling  

The goal is to build predictive models, evaluate performance, and compare their strengths.

---

## **Dataset**
- **Source:** `diabetes_prediction_dataset.csv`  
- **Number of records:** 100,000  

**Features:**

| Feature              | Type        | Description                                                                 |
|---------------------|------------|----------------------------------------------------------------------------|
| gender               | categorical | Patient gender                                                             |
| age                  | numeric     | Age in years                                                               |
| hypertension         | numeric     | Hypertension status (0 = No, 1 = Yes)                                      |
| heart_disease        | numeric     | Heart disease status (0 = No, 1 = Yes)                                     |
| smoking_history      | categorical | Smoking history (`ever`, `current`, `former`, `never`, `not current`, `No Info`) |
| bmi                  | numeric     | Body Mass Index                                                            |
| HbA1c_level          | numeric     | Blood sugar level over time                                                |
| blood_glucose_level  | numeric     | Current blood glucose                                                     |
| diabetes             | target      | 0 = No, 1 = Yes                                                           |

---

## **Data Cleaning & Preprocessing**
1. Replaced `"No Info"` in `smoking_history` with `NaN` and dropped those rows  
2. Removed duplicate rows (925 duplicates removed)  
3. One-hot encoded categorical variables: `gender` and `smoking_history`  
4. Train-test split: 80% training, 20% testing  
5. Scaling applied for KNN using `MinMaxScaler`  

```python
X = diabetes.drop("diabetes", axis=1)
y = diabetes["diabetes"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

- 
