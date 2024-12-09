import pandas as pd
from typing import List, Dict  
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from utils import cross_val_QWK
import optuna 
import json 

def objective(trial, reg : str, X : pd.DataFrame, y : pd.Series, num_cols : List[str], cat_cols : List[str], cv : int, verbose : bool=False) -> float:
    ''' 
    This function performs hyperparameter optimization using Optuna for a given classifier model.
    '''

    if reg == 'XGBoost':
        # Hyperparameters for XGBoost regressor
        params = {
            'objective' : trial.suggest_categorical('objective', ['reg:squarederror', 'reg:absoluteerror', 'reg:pseudohubererror']),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 500),
            'max_depth' : trial.suggest_int('max_depth', 5, 10),
            'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'gamma' : trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
            'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }

        model = XGBRegressor(**params)

    elif reg == 'LightGBM':
        # Hyperparameters for LightGBM regressor
        params = {
            'objective' : trial.suggest_categorical('objective', ['regression', 'poisson', 'quantile']),
            'verbosity' : -1,
            'n_estimators' : trial.suggest_int('n_estimators', 100, 500),
            'max_depth' : trial.suggest_int('max_depth', 5, 10),
            'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
            'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }

        model = LGBMRegressor(**params)

    elif reg == 'CatBoost':
        # Hyperparameters for CatBoost regressor
        params = {
            'objective' : trial.suggest_categorical('objective', ['RMSE', 'Poisson', 'Quantile']),
            'iterations': trial.suggest_int('iterations', 200, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }

        model = CatBoostRegressor(**params, verbose=0)
    
    elif reg == 'HistGradBoost':
        # Hyperparameters for HistGradBoost regressor
        params = {
            'loss' : trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'poisson']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_iter' : trial.suggest_int('max_iter', 100, 500),
            'max_leaf_nodes' : trial.suggest_int('max_leaf_nodes', 25, 50),
            'max_depth' : trial.suggest_int('max_depth', 5, 15),
            'l2_regularization' : trial.suggest_float('l2_regularization', 1, 5),
            'max_features' : trial.suggest_float('max_features', 0.5, 1.0)
        }

        model = HistGradientBoostingRegressor(**params)

    else:

        raise ValueError(f'Unsupported regressor type: {reg}')
    
    val_QWK = cross_val_QWK(model, X, y, num_cols, cat_cols, cv, optimize_mode=True)

    return val_QWK

def hyperparam_optim(reg : str, X : pd.DataFrame, y : pd.Series, num_cols : List[str], cat_cols : List[str], n_trials : int=50, cv : int=5) -> Dict[str, any]: 
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, reg, X, y, num_cols, cat_cols, cv),
        n_trials=n_trials 
    )

    print(f'Best parameters for {reg}: {study.best_params}')
    print(f'Best QWK score: {study.best_value}')

    with open(reg + '.txt', 'w') as f:
        json.dump(study.best_params, f)

    return study.best_params