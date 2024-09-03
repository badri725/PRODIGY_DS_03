# Bank Marketing Campaign Prediction using Decision Tree Classifier

This project aims to predict whether a client will subscribe to a term deposit based on the Bank Marketing Dataset. The dataset contains information from direct marketing campaigns of a Portuguese banking institution, and the classification goal is to predict if the client will subscribe (`yes`/`no`) to a term deposit.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Steps](#project-steps)
  - [1. Data Loading and Exploration](#1-data-loading-and-exploration)
  - [2. Data Cleaning and Preprocessing](#2-data-cleaning-and-preprocessing)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [4. Handling Outliers](#4-handling-outliers)
  - [5. Feature Selection](#5-feature-selection)
  - [6. Splitting the Data](#6-splitting-the-data)
  - [7. Building the Decision Tree Model](#7-building-the-decision-tree-model)
  - [8. Model Evaluation](#8-model-evaluation)
  - [9. Hyperparameter Tuning](#9-hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)
- [Next Steps](#next-steps)

## Project Overview

In this project, we:
1. Loaded and explored the dataset to understand its structure and contents.
2. Performed data cleaning, including handling duplicates and encoding categorical variables.
3. Conducted Exploratory Data Analysis (EDA) to visualize the distribution of features.
4. Removed outliers using the Interquartile Range (IQR) method.
5. Selected relevant features by analyzing correlations.
6. Built a Decision Tree Classifier to predict whether a client will subscribe to a term deposit.
7. Evaluated the model using metrics like accuracy, confusion matrix, and classification report.
8. Tuned the hyperparameters using GridSearchCV to improve the model's performance.

## Dataset

The dataset used in this project is the **Bank Marketing Dataset** from the UCI Machine Learning Repository. It is a 10% sample of the original dataset and contains 4,119 rows and 21 columns, including 20 features and 1 label (target).

## Installation

To run this project locally, you need to have Python installed along with the necessary libraries. You can install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Steps

### 1. Data Loading and Exploration

We start by loading the dataset and exploring its structure:

- Loaded the dataset using `pandas`.
- Renamed the target column from `y` to `deposit`.
- Displayed the first few rows, checked the shape, and reviewed summary statistics.

### 2. Data Cleaning and Preprocessing

- Checked for duplicates and missing values (none were found).
- Converted categorical columns to numerical values using `LabelEncoder`.

### 3. Exploratory Data Analysis (EDA)

- Visualized the distribution of numerical columns using histograms.
- Visualized the distribution of categorical columns using bar plots.

### 4. Handling Outliers

- Identified outliers in the `age`, `campaign`, and `duration` columns.
- Removed outliers using the Interquartile Range (IQR) method.

### 5. Feature Selection

- Calculated the correlation matrix to identify highly correlated features.
- Removed highly correlated features (`emp.var.rate`, `euribor3m`, `nr.employed`) to avoid multicollinearity.

### 6. Splitting the Data

- Split the data into training (75%) and testing (25%) sets using `train_test_split`.

### 7. Building the Decision Tree Model

- Built a Decision Tree Classifier using the Gini impurity criterion.
- Evaluated the model’s accuracy on the training and testing sets.

### 8. Model Evaluation

- Evaluated the model using confusion matrix, classification report, and accuracy score.
- Visualized the decision tree to understand how decisions are made.

### 9. Hyperparameter Tuning

- Performed hyperparameter tuning using `GridSearchCV` to find the best combination of `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- Evaluated the tuned model and visualized the optimized decision tree.

## Results

- The initial Decision Tree Classifier achieved an accuracy of approximately 90% on the test set.
- After hyperparameter tuning, the best model's accuracy improved slightly, and it was better suited for generalization on unseen data.

## Conclusion

The project successfully built a Decision Tree Classifier to predict whether a client will subscribe to a term deposit. The final model, after hyperparameter tuning, achieved a high accuracy, making it a robust tool for classification tasks in this dataset.

## Next Steps

- **Further Hyperparameter Tuning**: Experiment with more hyperparameters or finer grids.
- **Cross-Validation**: Implement k-fold cross-validation to ensure the model's robustness.
- **Model Comparison**: Compare the decision tree with other models like Random Forest, Logistic Regression, or SVM.
- **Feature Engineering**: Explore additional features that might improve model performance.
