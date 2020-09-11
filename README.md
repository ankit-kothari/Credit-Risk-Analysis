# Credit-Risk-Analysis
Predicting the ability of a borrower to pay back the loan through Traditional Machine Learning Models and comparing to Ensembling Methods 



## **Credit-Risk-Analysis**

Predicting the ability of a borrower to pay back the loan through a classification model using traditional machine learning Models and comparing to ensembling methods


### Traditional ML Model and  Ensembling techniques

[ankit-kothari/Credit-Risk-Analysis](https://github.com/ankit-kothari/Credit-Risk-Analysis/blob/master/credit_risk_analysis_ML.ipynb)

### Comparing XGBoost with ANN

[ankit-kothari/Credit-Risk-Analysis](https://github.com/ankit-kothari/Credit-Risk-Analysis/blob/master/credit_risk_analysis_Keras.ipynb)

[Data Overview ]
Some assumtions for the dataset:

| encoding | category | count of rows in the dataset
| --- | --- | --- | 
|1| Defaulted | 5634 | 
|0| Paid Back the amount in full | 33136 |


## Project

This project was aimed at exploring different traditional Machine Learning algorithms and comparing them against powerful models like ensembling methods and artificial neural networks in Keras to identify the credit risk and whether the customer will default or pay back the loan in full based on different indicators.

## Sampling

This dataset is an imbalanced dataset and so sampling was a must to get any good results otherwise the model will not be effective in figuring out False Negatives as they are a minority class and end up giving more bad loans.

```python
smote = SMOTE(ratio='minority')
features_c, target_c = smote.fit_sample(features_corr, target)
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
```

## Model Evaluation Criteria

- False positive rate is the number of false positives divided by the number of false positives plus the number of true negatives. This divides all the cases where we thought a loan would be paid off but it wasn't by all the loans that weren't paid off:

    fpr = fp / (fp + tn)

- True positive rate is the number of true positives divided by the number of true positives plus the number of false negatives. This divides all the cases where we thought a loan would be paid off and it was by all the loans that were paid off:

    tpr = tp / (tp + fn)

## Scaling and Normalizing the data

```python
def normalize(subset):
   continious_columns = subset.select_dtypes(include=['float']).columns
   mm_scaler = preprocessing.MinMaxScaler()
   for col in continious_columns:
     subset[col]= mm_scaler.fit_transform(subset[[col]])
   return subset
```

## Feature Engineering

### Correlation Matrix

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr1.png" width="60%">

## Model Architecture

### Logistic Regression with vanila

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr2.png" width="80%">

### Logistic Regression with Balanced weight penalty

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr3.png" width="80%">

### Logistic Regression with custom penalty

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr4.png" width="80%">

### Logistic Regression with SMOTE OVER SAMPLING:

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr5.png" width="80%">

### Logistic Regression with scaling and normalizing the data

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr6.png" width="80%">

### XGBOOST with scaling and normalizing the data with OVER SAMPLING

**XGBOOST HYPERPARAMETER**

- objective='binary:logistic' for binary classification and 'objective': 'multi:softmax' softmax for multiclass classification you also need to set num_class(number of classes)
- subsample=0.8 = subsample, which is for each tree the % of rows taken to build the tree.
- colsample_bytree: number of columns used by each tree.
- max_depth = It represents the depth of each tree, which is the maximum number of different features used in each tree.
- n_estimator = maximun number of decision tress.
- The booster parameter allows you to set the type of model you will use when building the ensemble. The default is gbtree which builds an ensemble of decision trees. If your data isn’t too complicated, you can go with the faster and simpler gblinear option which builds an ensemble of linear models.
- The gamma parameter can also help with controlling overfitting. It specifies the minimum reduction in the loss required to make a further partition on a leaf node of the tree.
- scoring: "f1" pr "accuracy"
- scale_pos_weight parameter impose greater penalties for errors on the minor class

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr7.png" width="80%">
### Modeling in Keras for Binary Classification  Using Under Sampling

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cr8.png" width="80%">

## Model  Performance

XGBOOST outperformed all the other algorithms and also was great in capturing False negatives with only 6 in a dataset of 20000 samples used for validation while also controling the False positives which were 2071. This model is great in detecting potential bad loans.

The ANN performed reasonably well too with an AUC of 0.87 as compared to 0.91 from XGBOOST, also the number of false negatives were higher using this model.

[Model Comparison ](https://www.notion.so/6d019d4624fe4499b0864fcf843d865f)


| Model Description | Sampling Method | AUC | 
| --- | --- | --- |
| LOGISTIC REGRESSSION VANILA | No Sampling | 0.50 |
| LOGISTIC REGRESSION: CLASS WEIGHT | Using sklearn class_weight="BALANCED" | 0.59 |
| LOGISTIC REGRESSION: WITH SAMPLING | SMOTE Over Sampling (minority class) | 0.58 |
| XGBOOST: BOOSTING | SMOTE-Over Sampling Method | 0.91 |
| MAX VOTING: BAGGING | SMOTE-Over Sampling Method | 0.80 |
| ANN-KERAS | SMOTE-Over Sampling Method | 0.87 |

