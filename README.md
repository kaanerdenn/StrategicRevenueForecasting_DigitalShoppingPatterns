# Strategic Revenue Forecasting & Digital Shopping Patterns

## Overview

This project aims to analyze online shopping data to predict customer purchasing intentions. Utilizing machine learning algorithms, the project identifies patterns and factors that influence whether a customer will make a purchase.

## Dataset

The dataset online_shoppers_intention.csv contains various features such as administrative duration, informational duration, product-related attributes, bounce rates, exit rates, page values, special days, and more, along with a target variable indicating the revenue outcome (purchase or no purchase).

## Features

Data Preprocessing: Cleaning and processing the dataset for analysis. This includes handling missing values and encoding categorical variables.
Exploratory Data Analysis (EDA): Utilizing Seaborn and Matplotlib for visualization to understand the distribution of the target variable and correlations between features.
Feature Engineering: Selecting relevant features and transforming them for better model performance.
Data Scaling and Transformation: Standardizing features using StandardScaler and applying PowerTransformer for normalization.
Machine Learning Modeling: Building and evaluating multiple models like Logistic Regression, Random Forest, CatBoost, and XGBoost classifiers.
Model Evaluation: Using metrics such as accuracy, confusion matrix, and classification reports to assess model performance.
Hyperparameter Tuning: Applying techniques like GridSearchCV for optimizing model parameters.
Requirements

This project requires Python and the following libraries:

Pandas
Numpy
Seaborn
Matplotlib
Scikit-learn
CatBoost
XGBoost
