# Credit-Risk-Analysis
Predicting the ability of a borrower to pay back the loan through Traditional Machine Learning Models and comparing to Ensembling Methods 

Some assumtions for the dataset:

| encoding | category |
| --- | --- |
|1| Defaulted |
|0| Paid Back the amount in full |


| Model Description | Sampling Method | AUC | 
| --- | --- | --- |
| LOGISTIC REGRESSSION VANILA | No Sampling | 0.50 |
| LOGISTIC REGRESSION: CLASS WEIGHT | Using sklearn class_weight="BALANCED" | 0.59 |
| LOGISTIC REGRESSION: WITH SAMPLING | SMOTE Over Sampling (minority class) | 0.58 |
| XGBOOST: BOOSTING | SMOTE-Over Sampling Method | 0.91 |
| MAX VOTING: BAGGING | SMOTE-Over Sampling Method | 0.80 |
| ANN-KERAS | SMOTE-Over Sampling Method | 0.87 |
