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
            
            model_report: dict = evaluate_models(X_train = X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
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