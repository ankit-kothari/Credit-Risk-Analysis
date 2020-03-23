# Credit-Risk-Analysis
Predicting the ability of a borrower to pay back the loan through Traditional Machine Learning Models and comparing to Ensembling Methods 

Some assumtions for the dataset:
|encoding | category |
|1:| Defaulted |
|0:| Paid Fine |

| Model Description | Sampling Method | AUC | 
| --- | --- |
| LOGISTIC REGRESSSION VANILA | List all new or modified files | List all new or modified files |
| LOGISTIC REGRESSION: CLASS WEIGHT | List all new or modified files | List all new or modified files |
| LOGISTIC REGRESSION: WITH SAMPLING | List all new or modified files | List all new or modified files |
| XGBOOST: BOOSTING |   | SMOTE-Over Sampling Method | .91
| MAX VOTING: BAGGING | List all new or modified files | SMOTE-Over Sampling Method | .80
| ANN-KERAS | Show file differences that haven't been staged | List all new or modified files |
