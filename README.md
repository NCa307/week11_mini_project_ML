# Diabetes Prediction Using Machine Learning

## **Project Overview**
This project predicts the presence of diabetes in patients based on health and lifestyle data using two machine learning models:

1. **Decision Tree Classifier** 
2. **K-Nearest Neighbors (KNN) Classifier** 

The goal is to explore predictive modeling, evaluate performance, and compare the strengths of each algorithm.
## **Dataset**
- **Source:** `diabetes_prediction_dataset.csv`  
- **Number of records:** 100,000  
- **Features (9 total):**  
| Feature               | Type        | Description                                                                      |
| --------------------- | ----------- | -------------------------------------------------------------------------------- |
| gender                | categorical | Patient gender                                                                   |
| age                   | numeric     | Age in years                                                                     |
| hypertension          | numeric     | Hypertension status (0 = No, 1 = Yes)                                            |
| heart\_disease        | numeric     | Heart disease status (0 = No, 1 = Yes)                                           |
| smoking\_history      | categorical | Smoking history (`ever`, `current`, `former`, `never`, `not current`, `No Info`) |
| bmi                   | numeric     | Body Mass Index                                                                  |
| HbA1c\_level          | numeric     | Blood sugar level over time                                                      |
| blood\_glucose\_level | numeric     | Current blood glucose                                                            |
| diabetes              | target      | 0 = No, 1 = Yes                                                                  |


**Data Cleaning Steps:**
1. Replace `"No Info"` in `smoking_history` with `NaN` and drop those rows  
2. Remove duplicate rows (925 duplicates removed)  
3. One-hot encode categorical variables: `gender` and `smoking_history`  

**Resulting dataset:** 63,259 rows × 13 columns

---

## **Data Preprocessing**
- **One-hot encoding:** converted categorical variables into dummy variables, dropping the first category to avoid multicollinearity  
- **Scaling (for KNN only):** normalized features to [0,1] using `MinMaxScaler`  
- **Train-test split:** 80% training, 20% testing
- 
