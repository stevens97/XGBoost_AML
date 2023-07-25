'''
# -----------------------
# Import Libraries
# -----------------------
'''

# OS
# -----------------------
import os

# Data Manipulation
# -----------------------

import pandas as pd
import numpy as np

# Data Visualisation
# -----------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
# -----------------------

# scikit-learn
# -----------------------

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb  # XGBoost
from xgboost import cv

'''
# -----------------------
# Main
# -----------------------
'''

if __name__ == "__main__":
    # Set path
    # -----------------------
    # Dataset: https: // www.kaggle.com / datasets / maryam1212 / money - laundering - data?resource = download
    df = pd.read_csv(r'C:\Users\cstevens\Desktop\Anti Money Laundering\archive\ML.csv')
    df.drop('isfraud', axis=1, inplace=True)
    df.describe()
    df.info()
    df.isnull().sum()
    df.dropna(inplace=True)

    # Feature Selection
    # -----------------------
    X = df.drop('typeoffraud', axis=1)
    y = df['typeoffraud']
    X.head()
    y.head()

    # Train-test Split
    # -----------------------
    X = pd.get_dummies(X).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    '''
    # Model Selection: XGBoost
    # -----------------------

    ## General Parameters:
    # -----------------------
    
    booster: The type of booster to use, e.g., gbtree (tree-based models) or gblinear (linear models).
    silent: Whether to print messages while running the model (default=1, set to 0 for verbose output).
    nthread: Number of threads to use for parallel processing.
    
    ## Tree Booster Parameters:
    # -----------------------
    
    eta or learning_rate: Step size shrinkage to prevent overfitting.
    min_child_weight: Minimum sum of instance weight needed in a child, used to control overfitting.
    max_depth: Maximum depth of a tree, controls the complexity of the model.
    gamma: Minimum loss reduction required to make a further partition on a leaf node.
    subsample: Subsample ratio of the training instances to prevent overfitting.
    colsample_bytree: Subsample ratio of columns when constructing each tree.
    lambda or reg_lambda: L2 regularization term on weights.
    alpha or reg_alpha: L1 regularization term on weights.
    scale_pos_weight: Control the balance of positive and negative weights in the dataset for imbalanced classes.
    
    ## Linear Booster Parameters:
    # -----------------------
    
    lambda or reg_lambda: L2 regularization term on weights.
    alpha or reg_alpha: L1 regularization term on weights.
    lambda_bias: L2 regularization term on the bias term.
    Learning Task Parameters:
    objective: The learning task and corresponding objective function (e.g., binary:logistic for binary classification).
    eval_metric: Evaluation metric used for early stopping and model evaluation (e.g., auc, error, logloss).
    
    ## Cross-validation Parameters:
    # -----------------------
    
    num_boost_round: The number of boosting rounds (trees) for training.
    early_stopping_rounds: Activates early stopping, based on a validation set's performance.
    evals: List of datasets to be used for early stopping.
    
    '''

    # Cross Validation
    # -----------------------

    classifier = xgb.XGBClassifier()

    params = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }

    rs_model = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='neg_log_loss',
                                  n_jobs=-1, cv=5, verbose=3)

    # Model Training
    # -----------------------

    rs_model.fit(X_train, y_train)

    # Model Evaluation
    # -----------------------
    y_pred = rs_model.best_estimator_.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
