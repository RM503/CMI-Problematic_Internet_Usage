import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

from main_funcs import cross_val_QWK
import json
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

folder_path = './child-mind-institute-problematic-internet-use/'

# Import files and merge data and actigraphy stats .csv files
train_df = pd.read_csv(folder_path + 'train.csv', index_col='id')
test_df = pd.read_csv(folder_path + 'test.csv', index_col='id')
# data_dict = pd.read_csv(folder_path + 'data_dictionary.csv')

train_timeseries = pd.read_csv('./actigraphy_stats/ts_train_stats.csv')
test_timeseries = pd.read_csv('./actigraphy_stats/ts_test_stats.csv')

train_df_merged = pd.merge(train_df, train_timeseries, how='left', on='id')
test_df_merged = pd.merge(test_df, test_timeseries, how='left', on='id')

train_df_merged.set_index('id', inplace=True)
test_df_merged.set_index('id', inplace=True)

train_cols = train_df_merged.columns.tolist()
test_cols = test_df_merged.columns.tolist()

# Remove sii label and PCIAT columns from training data
drop_cols = list( set(train_cols) - set(test_cols) )
drop_cols.remove('sii')

train_df_subset = train_df_merged[train_df.loc[:, 'sii'].notna()]
train_df_subset.drop(columns=drop_cols, inplace=True)

X_train = train_df_subset.loc[:, train_df_subset.columns != 'sii']
y = train_df_subset['sii'].astype('Int64')
X_test = test_df_merged 

# Define binned-age
train_df['Basic_Demos-Age_binned'] = pd.cut(
    train_df['Basic_Demos-Age'], bins=[4, 11, 17, 25], labels=[0, 1, 2]
).astype('int32')

def extra_features(df : pd.DataFrame) -> pd.DataFrame:
    # binned-age
    df['Basic_Demos-Age'] = train_df['Basic_Demos-Age_binned']
    df.rename(columns={'Basic_Demos-Age':'Basic_Demos-Age_binned'}, inplace=True)

    # fat to BMI ratio
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']

    # fat-free and fat mass indices to BMI ratio
    df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
    df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']

    # muscle to fat ratio
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']

    # body water to weight ratio
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']

    return df

X_train = extra_features(X_train)
X_test = extra_features(X_test)

if __name__ == '__main__':
    
    with open('./final_features/final_features.txt', 'r') as f:
        features = json.load(f)

    numerical_cols = features['numerical']
    categorical_cols = features['categorical']

    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.019560871979228026,
        'gamma': 2.8602177174948795,
        'reg_alpha': 3.350008536536844,
        'reg_lambda': 2.135446274120378,
        'subsample': 0.9514354733634935,
        'colsample_bytree': 0.8922119852674855
    }

    lgb_params = {
        'objective': 'regression',
        'n_estimators': 275,
        'max_depth': 5,
        'learning_rate': 0.02344410451744192,
        'reg_alpha': 4.771123633686727,
        'reg_lambda': 2.108926126425786,
        'subsample': 0.6734397314373017,
        'colsample_bytree': 0.7205525456429738
    }

    cb_params = {
        'objective': 'RMSE',
        'iterations': 339,
        'depth': 8,
        'learning_rate': 0.052883943723149085,
        'l2_leaf_reg': 4.689002920221734,
        'subsample': 0.7739756088803678
    }

    RANDOM_SEED = 10

    xgb_reg = XGBRegressor(**xgb_params, random_state=RANDOM_SEED)
    lgb_reg = LGBMRegressor(**lgb_params, verbosity=-1, random_state=RANDOM_SEED)
    cb_reg = CatBoostRegressor(**cb_params, verbose=0, random_state=RANDOM_SEED)

    voting_reg = VotingRegressor(
        estimators=[
            ('XGBoost', xgb_reg),
            ('LightGBM', lgb_reg),
            ('CatBoost', cb_reg)
        ],
        weights = [2, 1, 3]
    )

    y_val, y_val_pred, y_pred = cross_val_QWK(voting_reg, X_train, X_test, y, numerical_cols, categorical_cols, cv=5, verbose=True)