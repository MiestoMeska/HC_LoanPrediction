<p align="center">
    <img src="https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/assets/Home-Credit-logo.png" alt="Home Credit Logo" width="50%">
</p>

# Home Credit Group Loan Risk Prediction


## Introduction

Welcome to the project of Home Credit data analysis and machine learning techniques application. In this project I am diving into Home Credit's extensive dataset to understand creditworthiness factors and build a predictive model. After developing the model, accuracy will be tested by submitting predictions to Kaggle for evaluation.

## Project Information

[Dataset Structure Notebook](https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/notebooks/dataset_composition.ipynb)

This notebook is about exploring the initial structure and composition of the Home Credit dataset. It details the different data files, their sizes, types, and relationships. It aims to provide a clear understanding of the dataset's framework necessary for subsequent data manipulation and analysis.

[EDA Notebook](https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/notebooks/train_dataset_EDA.ipynb)

Exploratory Data Analysis (EDA) on the Home Credit training dataset. It includes visualizations of key variables, investigations into missing values, and analyses of relationships between features and the target variable. The objective is to uncover insights and patterns that inform future steps in feature engineering and modeling.


[Feature Engineering Notebook](https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/notebooks/feature_engineering.ipynb)

This notebook is about enhancing the dataset with new features derived from existing data.


[Feature Selection Notebook](https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/notebooks/feature_selection.ipynb)

The Feature Selection Notebook is about iteratively analyzing feature importance and reducing the number of features used for model training. This involves identifying less significant features and removing them to improve the efficiency and performance of the predictive models. The process aims to retain only the most impactful attributes that contribute to the accuracy of creditworthiness predictions.


[Model Training Notebook](https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/notebooks/model_training_final.ipynb)


This notebook is about applying a machine learning technique to develop and evaluate a predictive model. It covers hyperparameter tuning and model validation strategies, focusing on training the best-performing model. The primary goal is to prepare and optimize the model for generating predictions, which will be submitted to Kaggle for evaluation.

## Model Performance

<img src="https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/assets/ROC.PNG" alt="ROC model performance" width="65%">

The validation results indicate that the predictive model has a moderate level of accuracy and a decent ROC AUC score, suggesting a satisfactory ability to distinguish between classes. However, the precision is low, indicating a higher rate of false positives. The recall rate is relatively high, suggesting the model is quite good at identifying the positive class but at the cost of incorrectly labeling many negatives as positives. Overall, while the model shows promise in identifying positive cases, it requires refinement to reduce false positives and improve overall precision.

Kaggle submission score:

![Kaggle Score](https://github.com/MiestoMeska/HC_LoanPrediction/blob/main/assets/submission_score.PNG "Kaggle score")

## Conclusions

The loan default prediction model effectively identifies most actual defaults, which is essential for minimizing financial risks. However, it tends to misclassify some non-defaulters as defaulters, leading to unnecessary actions. The next steps should focus on improving the precision to reduce false positives, while maintaining the strong detection rate of true defaults.
