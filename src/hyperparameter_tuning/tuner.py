import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
import joblib
from skopt import gp_minimize
from config import paths
from logger import get_logger
import os
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from config.paths import OPT_HPT_DIR_PATH
from xgboost import XGBClassifier
from sklearn.svm import SVC

np.int = np.int_


HPT_RESULTS_FILE_NAME = "HPT_results.csv"

logger = get_logger(task_name="tune")


"""
    The following fucntion used sickit-optimization module for hyper parameter tuning of models.
    We need to define the start and end range of integer or float (Real) hyper-parameters. This defined space 
    is then used in cross fold validation for each model hyper parameter tuning.
"""


def run_hyperparameter_tuning(train_X , train_Y):

    svc_space  = [
              Real(1.0, 5.0, name='C'),
              Integer(2, 10, name='degree'),
              Real(0.0,2.0, name = 'coef0'),
              Real(0.001,0.009, name = 'tol')
             ]

    rf_space  = [
              Integer(5, 120, name='n_estimators'),
              Integer(1, 10, name='max_depth'),
              Integer(2,20, name = 'min_samples_split'),
              Integer(1,10, name = 'min_samples_leaf')
             ]
    
    xgb_space  = [
              Integer(1,50, name='min_child_weight'),
              Integer(2, 120, name='xg_n_estimators'),
              Integer(0,50, name = 'gamma')
             ]
    
    svc = SVC()
    
    rf = RandomForestClassifier(max_features="log2")
   
    xgb = XGBClassifier()
    

    @use_named_args(rf_space)
    def objective_rf(**params):
        rf.set_params(**params)
        return -np.mean(cross_val_score(rf, train_X, train_Y, cv=10, n_jobs=-1,scoring="f1"))

    @use_named_args(svc_space)
    def objective_svc(**params):
        svc.set_params(**params)
        return -np.mean(cross_val_score(svc, train_X, train_Y, cv=10, n_jobs=-1,scoring="f1"))
    
    @use_named_args(xgb_space)
    def objective_xgb(**params):
        xgb.set_params(**params)
        return -np.mean(cross_val_score(xgb, train_X, train_Y, cv=10, n_jobs=-1,scoring="f1"))

    svc_gp = gp_minimize(objective_svc, svc_space, n_calls=100, random_state=42)
    res_gp = gp_minimize(objective_rf, rf_space, n_calls=100, random_state=42)
    xgb_gp = gp_minimize(objective_xgb, xgb_space, n_calls=100, random_state=42)

    

   
    best_hyperparameters = {
            "C": svc_gp.x[0],
            "degree": svc_gp.x[1],
            "coef0": svc_gp.x[2],
            "tol":  svc_gp.x[3],
           
        
            "rf_n_estimators":res_gp.x[0] , 
            "rf_max_depth":res_gp.x[1] ,
            "min_samples_split":res_gp.x[2] , 
            "min_samples_leaf":res_gp.x[3] , 
            "max_features":"log2",

            "min_child_weight":  xgb_gp.x[0],
            "xg_n_estimators":  xgb_gp.x[1],
            "gamma": xgb_gp.x[2],
        }
    
    # Making data hyper paramters directory
    if not os.path.exists(paths.OPT_HPT_DIR_PATH):
        os.makedirs(paths.OPT_HPT_DIR_PATH)
    
    joblib.dump(best_hyperparameters,OPT_HPT_DIR_PATH+"/optimized_hyper_parameters.joblib")
    return best_hyperparameters

    
    




