# -Predicting-Hospital-Admission-Price-By-Implementing-Machine-Learning

Project Member: 

![Hospital Cover](https://github.com/Stranger-Descendant/-Predicting-Hospital-Admission-Price-By-Implementing-Machine-Learning/raw/main/hospital%20cover.jpeg)


## Table of Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Baseline Model](#Baseline-Model)
- [Optimization Model](#Optimization-Model)
- [Results and Analysis](#results-and-analysis)
- [Business Insights](#Business-Insights)
- [References](#References)

## Introduction

This project aims to implement advanced machine learning techniques to accurately predict hospital billing costs for patients. By analyzing various patient data and health metrics, the model seeks to enhance financial planning and decision-making for healthcare providers.

## Data

The dataset used consists of 2,772 entries across 7 columns, focusing on individual medical costs billed by health insurance. Key variables include age, sex, BMI, number of children, smoking status, region, and charges. The dataset contains categorical features (e.g., sex, smoker, region) and continuous variables (e.g., age, BMI, charges).

## Exploratory Data Analysis

- Descriptive Statistics: Initial analysis reveals insights into demographics, showing a higher prevalence of younger individuals, lower smoking rates, and an even distribution across regions.
- Visualizations: Various plots (histograms, box plots, and heatmaps) illustrate the distribution of demographic characteristics and their correlation with medical charges, highlighting significant relationships between smoking status, age, BMI, and charges.

## Baseline Model

- Preprocessing: Categorical features are encoded using one-hot encoding, while numerical features remain unchanged. The dataset is split into training and testing sets for model evaluation.
- Model Selection: Several regression models are implemented, including Linear Regression, Random Forest, and XGBoost, to predict continuous insurance charges.


## Optimization Model

- Training and Evaluation: Models are trained, and metrics such as R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE) are calculated to assess their performance. The "Extra Trees Regression" model emerges as the best performer.
- Classification Models: The continuous charges variable is transformed into a binary category (above/below median), allowing for classification. Logistic Regression and XGBoost Classification yield the highest accuracies.

## Results and Analysis

- Confusion Matrix
lorem ipsum

- ROC Curve Analysis: A comparison of ROC curves for different classifiers demonstrates strong performance, particularly for Logistic Regression and XGBoost, with high area under the curve (AUC) scores.
![ROC Curve](https://github.com/Stranger-Descendant/-Predicting-Hospital-Admission-Price-By-Implementing-Machine-Learning/raw/main/ROC.png)


## Business Insights

The findings emphasize the critical role of lifestyle choices (e.g., smoking status) in determining health insurance premiums. These insights can inform healthcare providers about the financial implications of various patient demographics and encourage targeted interventions for cost management.
