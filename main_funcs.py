'''
This is a module file containing a list of important functions. 

The cross_val_QK() function takes in a number of arguments as inputs and serves two important purposes. If optimize_mode=True, the function
is used to perform hyperparameter optimization, in conjuction with Optuna, and returns the optimized QWK score. If optimize_mode=False, it
means that optimization has already been performed and returns arrays with validation and prediction results.
'''

import numpy as np
import pandas as pd
from typing import List, Union, Tuple 
from scipy.optimize import minimize
from sklearn.base import clone 
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import cohen_kappa_score, f1_score

RANDOM_SEED = 10

def clipped(X : pd.DataFrame, r : float) -> pd.DataFrame:
    ''' 
    This function caps the numerical columns to the specified upper and lower quantiles on respective ends. It does so for those featutres
    for which the standard deviation is much greater or equal the mean. This is passed as a desired ratio of std/mean.

    The function returns a dataframe with the desired level of clippining.
    '''
    X_clipped = X.copy()

    ratio = X_clipped.std() / X_clipped.mean()
    idx_list = ratio[ ratio.where(ratio >= r).notna() ].index.tolist()

    for idx in idx_list:
        lq = X_clipped[idx].quantile(0.05)
        uq = X_clipped[idx].quantile(0.95)
        X_clipped.loc[:, idx] = X_clipped.loc[:, idx].clip(lower=lq, upper=uq)

    return X_clipped

def process(X : pd.DataFrame, num_cols : List[str], cat_cols : List[str]) -> pd.DataFrame:
    ''' 
    This function performs preprocessing on the data. The training data is first split into numerical and categorical features depending 
    on the numerical and categorical columns passed as lists.

    The function returns a dataframe containing only the selected numerical and categorical columns.
    '''
    X_numerical = X[num_cols]

    X_categorical = X[cat_cols]
    X_categorical = X_categorical.astype('Int64')

    # Some numerical features have erroneous zero values - e.g. BMI, blood pressure etc
    # Winsorize X_numerical and replace occurrences of zero with feature mean

    X_numerical = clipped(X_numerical, 1.5)
    X_numerical.replace(0, X_numerical.mean(axis=0), inplace=True)

    X_processed = X_categorical.join(X_numerical)

    return X_processed

def quadratic_weighted_kappa(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_rounder(oof_not_rounded : np.ndarray, thresholds : List[float]) -> np.ndarray:
    ''' 
    This function iteratively rounds up (or down) the validation set predictions using a predefined threshold passed as an array.
    By default, the thresholds can be set as [0.5, 1.5, 2.5], such that predicts lesser (or greater) than the thresholds will be
    rounded to the nearest integer.
    
    This function can be particularly useful when one wishes to implement a custom rounding threshold. If optimize_mode=True, the function
    is used to perform hyperparameter optimization, in conjuction with Optuna, and returns the optimized QWK score. If optimize_mode=False, it
    means that optimization has already been performed and returns arrays with validation and prediction results.
    '''
    thresh_0, thresh_1, thresh_2 = thresholds # unpack the thresholds

    return np.where(
        oof_not_rounded < thresh_0, 0, np.where(oof_not_rounded < thresh_1, 1, np.where(oof_not_rounded < thresh_2, 2, 3))
    )

def evaluate_predictions(thresholds : List[float], y : np.ndarray, oof_not_rounded : np.ndarray) -> float:
	y_pred_rounded = threshold_rounder(oof_not_rounded, thresholds)
	return -quadratic_weighted_kappa(y, y_pred_rounded)

def cross_val_QWK(reg_class, X : pd.DataFrame,
                  X_test : pd.DataFrame, y : pd.Series, num_cols : List[str], cat_cols : List[str], cv : int=5,
                  verbose : bool = False, optimize_mode : bool = False) -> Union[float, Tuple[pd.Series, np.ndarray, np.ndarray]]:
    ''' 
    This function takes in a particular classifier and training data and perfoms Stratified k-Fold cross-validation on it
    and calculates the out-of-fold (OOF) QWK score.

    The output depends on whether optimize_mode is True or False. If True, it only returns the tuned value of kappa. This can be important during hyperparameter optimization when
    other outputs are not required.
    '''
    X = process(X, num_cols, cat_cols) 
    N_SPLITS = cv
    train_scores = [] # training QWK scores across folds
    val_scores = []   # validation QWK scores across folds

    oof_not_rounded = np.zeros(len(y), dtype=np.float32) # array for storing out-of-fold prediction from regressor
    oof_rounded = np.zeros(len(y), dtype=np.int32) # array for storing out-of-fold prediction that has been rounded
    y_pred = np.zeros((len(X_test), N_SPLITS))
    
    SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    ''' 
    Performing Stratified k-fold cross validation with n_splits=5; the training set XX will be further
    split into training and validation sets a total of n_split times. The data preprocessing steps (imputation and scaling)
    will be applied here to prevent data leakage.
    '''
    for fold, (train_idx, test_idx) in enumerate(SKF.split(X, y)):

        # Breaking up the training data into further training and validation sets during each iteration
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        # Data pre-processing

        numerical_transformer = Pipeline(
            steps=[
                #('imp_num', KNNImputer(n_neighbors=5, weights='uniform')),
                ('imp_num', SimpleImputer(strategy='mean')),
                #('ss', StandardScaler()),
                ('mms', MinMaxScaler()),
                #('rs', RobustScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                #('imp_cat', KNNImputer(n_neighbors=5, weights='uniform')),
                ('imp_cat', SimpleImputer(strategy='most_frequent')),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical_transforms', numerical_transformer, num_cols),
                ('categorical_transforms', categorical_transformer, cat_cols)
            ]
        )

        preprocessor.set_output(transform='pandas')
        X_train = preprocessor.fit_transform(X_train) 
        X_val = preprocessor.transform(X_val) 
        test_data = preprocessor.transform(X_test) 

        reg = clone(reg_class)
        reg.fit(X_train, y_train)

        y_train_pred = reg.predict(X_train)
        y_val_pred = reg.predict(X_val)

        oof_not_rounded[test_idx] = y_val_pred
        oof_rounded[test_idx] = y_val_pred.round(0).astype('int32')

        kappa_train = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype('int32'))
        kappa_val = quadratic_weighted_kappa(y_val, y_val_pred.round(0).astype('int32'))

        train_scores.append(kappa_train)
        val_scores.append(kappa_val)
        y_pred[:, fold] = reg.predict(test_data)

        if verbose:

            print(f'Fold {fold + 1}: Training QWK = {kappa_train:.4f}, Validation QWK: {kappa_val:.4f}')

    if verbose:

        print(f'Mean training QWK = {np.mean(train_scores):.4f}')
        print(f'Mean validation QWK = {np.mean(val_scores):.4f}')

    ''' 
    In kappa_optimizer the QWK metric is maximized (or rather the negative QWK is minimize) to obtain a set of optimal 
    rounding threshold values.
    '''
    kappa_optimizer = minimize(evaluate_predictions, x0 = [0.5, 1.5, 2.5], args=(y, oof_not_rounded), method='Nelder-Mead')
    threshold_optim = kappa_optimizer.x
    #assert kappa_optimizer.succes, 'Optimizer did not converge.'

    oof_tuned = threshold_rounder(oof_not_rounded, threshold_optim)
    kappa_val_tuned = quadratic_weighted_kappa(y, oof_tuned)

    F1_score = f1_score(y[test_idx].to_numpy(), oof_tuned[test_idx], average='weighted')

    print(f'Tuned QWK = {kappa_val_tuned}; F1 score = {F1_score}')

    y_pred_mean = threshold_rounder(y_pred.mean(axis=1), threshold_optim)

    if optimize_mode == True:

        return kappa_val_tuned
    else:

        return y[test_idx], oof_tuned[test_idx], y_pred_mean 