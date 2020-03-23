# Credit-Risk-Analysis
Predicting the ability of a borrower to pay back the loan through Traditional Machine Learning Models and comparing to Ensembling Methods 

Some assumtions for the dataset:

| encoding | category | count of rows in the dataset
| --- | --- | --- | 
|1| Defaulted | 5634 | 
|0| Paid Back the amount in full | 33136 |


| Model Description | Sampling Method | AUC | 
| --- | --- | --- |
| LOGISTIC REGRESSSION VANILA | No Sampling | 0.50 |
| LOGISTIC REGRESSION: CLASS WEIGHT | Using sklearn class_weight="BALANCED" | 0.59 |
| LOGISTIC REGRESSION: WITH SAMPLING | SMOTE Over Sampling (minority class) | 0.58 |
| XGBOOST: BOOSTING | SMOTE-Over Sampling Method | 0.91 |
| MAX VOTING: BAGGING | SMOTE-Over Sampling Method | 0.80 |
| ANN-KERAS | SMOTE-Over Sampling Method | 0.87 |

*PROJECT*

This project was aimed at exploring different traditional Machine Learning algorithms and comparing them against powerful models and ensembling methods and artificial nueral network in Keras.

*DATASET*

This dataset is a imbalanced dataset and so sampling was a must to get any good results othwerwise model will not be effective in figuring out False Negatives as they are a minority class and end up giving more bad loans. 

*MODEL PERFORMANCE*

XGBOOST outperformed all the other algorithms and also was great in capturing False negatives with only 6 in a dataset of 20000 samples used for validation while also controling the False positives which were 2071. This model is great in detecting potential bad loans. 

The ANN performed reasonably well too with an AUC of 0.87  as compared to 0.91 from XGBOOST, also the number of false negatives were higher using this model.
