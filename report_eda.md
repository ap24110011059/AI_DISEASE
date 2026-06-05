# EDA Report – Pima Indians Diabetes Dataset

## Data Source

The dataset used in this project is the Pima Indians Diabetes Dataset. It contains medical information collected from female patients of Pima Indian heritage and is commonly used for diabetes prediction research.

Dataset Size:

* 768 records
* 8 input features
* 1 target variable (Outcome)

Target Variable:

* 0 = Non-Diabetic
* 1 = Diabetic

---

## Class Balance

The dataset is not perfectly balanced.

Class Distribution:

* Non-Diabetic (Outcome = 0): 500 patients
* Diabetic (Outcome = 1): 268 patients

This class imbalance may affect machine learning model performance and should be considered during model training and evaluation.

---

## Missing Value Strategy

Some medical features contained zero values that are not realistic and were treated as missing values.

Affected Features:

* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI

Strategy Used:

* Replaced zero values with the median of the respective column.
* Median imputation was chosen because it is less sensitive to outliers than the mean.

---

## Feature Distributions

Histograms were generated for all features to understand their distributions.

Observations:

* Glucose values showed a wide range and appeared to be an important predictor of diabetes.
* BMI values were concentrated around the overweight and obese ranges.
* Age distribution was skewed toward younger and middle-aged patients.
* Insulin values showed high variability with several extreme values.
* Pregnancies had a right-skewed distribution.

---

## Key Insights

### 1. Class Imbalance Exists

The dataset contains significantly more non-diabetic patients than diabetic patients (500 vs 268). This imbalance can influence classification performance and may require special handling techniques.

### 2. Glucose Appears Highly Informative

Patients with higher glucose levels are more likely to belong to the diabetic class. Glucose is expected to be one of the strongest predictive features.

### 3. BMI and Age May Influence Diabetes Risk

Higher BMI values and increasing age appear to be associated with a greater likelihood of diabetes, suggesting their importance in predictive modeling.

---

## Conclusion

Exploratory Data Analysis helped identify important dataset characteristics including class imbalance, missing values, and feature behavior. These findings provide a strong foundation for preprocessing, feature engineering, and machine learning model development.
