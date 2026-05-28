# Summary of Paper
## Breast Cancer Recurrence Prediction Using Machine Learning

## Problem

* Breast cancer is one of the most commonly diagnosed cancers among women worldwide, and recurrence after treatment is still a major issue.
* Around 30% of patients may develop recurrence even after receiving proper treatment in the early stages.
* In many cases, recurrence is identified only after visible symptoms appear, making early treatment difficult.
* Existing prediction models like PREDICT mainly depend on factors such as tumor size, lymph node involvement, and tumor grade. These factors alone cannot fully describe the complex nature of breast cancer recurrence.
* Most machine learning studies focus on imaging and pathological data, but these are not always easily available in every hospital setting.
* Because of this, researchers aimed to develop a practical prediction model using routine clinical information and laboratory test results that are easily available from hospital records.

## Method

* The study included data from 342 breast cancer patients treated at Tianjin Medical University Cancer Institute and Hospital between 2011 and 2018.
* Researchers collected 25 clinical and laboratory features from electronic medical records.

### Important clinical features included:

* Age
* Menopause status
* Tumor size
* Histological grade
* Lymph node stage
* Molecular subtype
* Treatment strategy

### Important laboratory biomarkers included:

* CA125

* CEA

* CA15-3

* Fibrinogen (Fbg)

* D-Dimer

* Coagulation-related indicators

* Eleven machine learning algorithms were compared in the study:

  * Logistic Regression (LR)
  * Random Forest (RF)
  * Support Vector Classification (SVC)
  * XGBoost
  * Gradient Boosting Decision Tree (GBDT)
  * Decision Tree
  * Multilayer Perceptron (MLP)
  * Linear Discriminant Analysis (LDA)
  * AdaBoost
  * Gaussian Naive Bayes (GaussianNB)
  * LightGBM

* The dataset was divided into:

  * 70% training data
  * 30% testing data

* A 3-fold cross-validation method was used during training to improve model reliability.

* SHAP analysis was used to identify the importance of each feature, while Decision Curve Analysis (DCA) was used to evaluate clinical usefulness.

## Key Results

* Among all the machine learning models, AdaBoost showed the best overall performance in predicting breast cancer recurrence.

### AdaBoost Performance:

* AUC: 0.987

* Accuracy: 97.1%

* Sensitivity: 94.7%

* Specificity: 97.6%

* Positive Predictive Value (PPV): 90.0%

* Negative Predictive Value (NPV): 98.8%

* F1 Score: 0.923

* SHAP analysis showed that the most important factors influencing recurrence prediction were:

  * CA125
  * CEA
  * Fibrinogen (Fbg)
  * Tumor diameter

* Higher levels of CA125, CA15-3, D-Dimer, and fibrinogen were associated with a higher recurrence risk.

* The study also found that molecular subtype and lymph node involvement strongly affected recurrence probability.

* Decision Curve Analysis proved that AdaBoost provided better clinical benefit compared to the other machine learning algorithms.

## Gap Identified

* Although the results were highly accurate, the study still has some limitations.
* The sample size was relatively small, with only 342 patients included.
* External validation using patients from different hospitals was not performed.
* Since the study was retrospective, there is a possibility of selection bias.
* Genetic mutation and advanced molecular data were not included in the prediction model, even though these factors can influence recurrence.
* The study also did not clearly differentiate between early recurrence and late recurrence cases.
* Future studies using larger datasets and more biological information may improve the prediction system further.

## Conclusion

* The study successfully developed a machine learning-based prediction model for breast cancer recurrence using routine clinical and laboratory data.
* AdaBoost achieved the highest prediction accuracy among all algorithms tested.
* The model may help doctors identify high-risk patients earlier and support better treatment planning and follow-up care.
