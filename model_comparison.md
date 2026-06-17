# Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.7078 | 0.6047 | 0.4815 | 0.5361 | 0.8067 |
| DT | 0.6883 | 0.6364 | 0.2593 | 0.3684 | 0.7641 |
| RF | 0.7727 | 0.7111 | 0.5926 | 0.6465 | 0.8170 |
| SVM | 0.7079 | 0.6047 | 0.4815 | 0.5361 | 0.8165 |
| MLP | 0.7208 | 0.6122 | 0.5556 | 0.5825 | 0.6828 |

## Model Analysis

Among all the models evaluated, Random Forest achieved the highest F1 Score of 0.6465. The F1 Score is important because it balances both precision and recall, making it suitable for healthcare datasets where class imbalance exists. Random Forest performed better because it combines multiple Decision Trees, reducing overfitting and improving generalization. As a result, it was able to identify diabetic patients more effectively while maintaining good prediction accuracy, making it the strongest baseline model in our study.