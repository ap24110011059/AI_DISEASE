# Gap Analysis – Early Disease Detection Using Machine Learning

## Paper Reviewed

**An Experimental Study for Early Diagnosing Parkinson's Disease Using Machine Learning (2023)**. The study focused on early detection of Parkinson's Disease using machine learning techniques applied to clinical characteristics, voice features, and motor examination data. The researchers used preprocessing techniques such as MinMax Scaling, outlier removal, and SMOTE balancing before training multiple machine learning models. The study reported high classification accuracy for identifying Parkinson's Disease and high-risk patients.

---

## What the Paper Does

The paper aims to improve early diagnosis of Parkinson's Disease before severe symptoms appear. Early detection is important because treatment is more effective during the early stages of the disease.

Main contributions include:

* Collection and preprocessing of clinical and voice-based features.
* Use of machine learning algorithms for disease classification.
* Application of MinMax Scaling and SMOTE for preprocessing.
* Prevention of data leakage and overfitting during model development.
* Evaluation of model performance using classification metrics.

---

## What Is Missing

Although the study achieved strong performance, several limitations remain:

1. The dataset was relatively small and may not represent larger populations.
2. The study focused only on Parkinson's Disease rather than creating a generalized disease prediction framework.
3. External validation on independent datasets was limited.
4. The model relied on specialized clinical and voice-related features that may not be available in all healthcare settings.
5. The paper focused mainly on prediction accuracy and less on model interpretability and deployment in routine healthcare environments.

---

## How Our Work Fills the Gap

Our project focuses on building a practical and reproducible disease prediction pipeline using commonly available healthcare data.

Our contributions include:

* Using structured clinical datasets that are easier to obtain.
* Performing exploratory data analysis to understand data quality and feature behavior.
* Handling missing values through median imputation.
* Applying normalization and reproducible train-test splitting strategies.
* Comparing multiple machine learning models including Logistic Regression, Decision Tree, Random Forest, and SVM.
* Evaluating models using Accuracy, Precision, Recall, F1-Score, and AUC-ROC rather than relying on a single metric.
* Creating a reusable pipeline that can be extended to additional diseases in future research.

---

## Future Improvements

Future work may include:

* Testing on larger multi-hospital datasets.
* Incorporating explainable AI techniques such as SHAP.
* Exploring adaptive and continual learning models.
* Integrating wearable sensor and real-time health monitoring data.
* Validating the models on multiple disease prediction tasks.

---

## Conclusion

The reviewed paper demonstrates the potential of machine learning for early disease detection. However, challenges related to dataset size, generalization, and practical deployment remain. Our work addresses these issues by developing a reproducible machine learning pipeline, evaluating multiple algorithms, and focusing on methods that can be applied to real-world healthcare datasets.