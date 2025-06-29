import os 
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Support Vector Classifier": SVC(),
                "XGBClassifier": XGBClassifier()
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [7],
                    'criterion': ['entropy'],
                    'random_state': [7]
                },
                "Logistic Regression": {
                    'max_iter': [1000]
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3]
                },
                "Support Vector Classifier": {
                    # Default parameters as used in notebook
                },
                "XGBClassifier": {
                    'eval_metric': ['logloss'],
                    'max_depth': [3],
                    'learning_rate': [0.2],
                    'n_estimators': [180],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'reg_alpha': [0.3],
                    'reg_lambda': [1],
                    'random_state': [42]
                }
            }
            
            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            
        except Exception as e:
            raise CustomException(e, sys)
        

