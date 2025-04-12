import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Random Forest' : RandomForestRegressor(),
                'Linear Regression' : LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'K Neighbors Regressor' : KNeighborsRegressor(),
                'Decision Tree Regressor' : DecisionTreeRegressor(),
                'Adaboost Regressor' : AdaBoostRegressor(),
                'XGBoost Regressor' : XGBRegressor()
             }
            
            params = {
                "LinearRegression": {
                    # No hyperparameters to tune usually
                    "fit_intercept": [True, False],
                    "normalize": [True, False]  # Deprecated in newer versions, used only if needed
                },

                "Ridge": {
                    "alpha": [0.01, 0.1, 1, 10, 100],
                    "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sag"]
                },

                "Lasso": {
                    "alpha": [0.01, 0.1, 1, 10],
                    "fit_intercept": [True, False],
                    "selection": ["cyclic", "random"]
                },

                "ElasticNet": {
                    "alpha": [0.01, 0.1, 1],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "fit_intercept": [True, False]
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]
                },

                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2, 4]
                },

                "RandomForestRegressor": {
                    "n_estimators": [100, 200],
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["auto", "sqrt", "log2"]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.5, 0.8, 1.0],
                    "colsample_bytree": [0.5, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.3],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [1, 1.5, 2]
                },

            }

            
            model_report: dict = evaluate_models(X_train = X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            ## Getting best model
            best_model_score = max(sorted(model_report.values()))
            
            ## To get best model score from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6 :
                raise CustomException("No best model found")
            logging.info(f"Best model on both training and testing dataset")
            
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_scor = r2_score(y_test,predicted)
            
            return r2_scor
            
        except Exception as e:
            raise CustomException(e,sys)