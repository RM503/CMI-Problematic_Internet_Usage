import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

class time_series_features:
    ''' 
    This class contains methods for extracting statistics from the actigraphy parquet files. The class can be instantiated with only one argument
    where 'file_path_array' is an array containing the file paths of all the parquet files.

    If 'extract_timeseries_features()' and/or 'process_file()' methods are to be used standalone, then an extra parameter is required.
    '''
    def __init__(self, filepath_array):
        self.filepath_array = filepath_array 

    def extract_timeseries_features(self, df):
        ''' 
        Extracting time-series features from the actigraphy datasets.
        Copied from Lennard Haupt's notebook in https://www.kaggle.com/code/lennarthaupts/cmi-detecting-problematic-digital-behavior
        '''

        # Converting time_of_day to hours; the time_of_day variable is expressed in nanoseconds
        df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)

        # Define conditions for night, day, and no mask (full data); night is defined to be between hours 22 and 5
        night = ((df["hours"] >= 22) | (df["hours"] <= 5))
        day = ((df["hours"] <= 20) & (df["hours"] >= 7))
        no_mask = np.ones(len(df), dtype=bool)
        # Define weekend and last week conditions
        weekend = (df["weekday"] >= 6)
        last_week = df["relative_date_PCIAT"] >= (df["relative_date_PCIAT"].max() - 7)
        # Create additional weekend features
        df["enmo_weekend"] = df["enmo"].where(weekend)
        df["anglez_weekend"] = df["anglez"].where(weekend)
        # Basic features 
        features = [
            df["non-wear_flag"].mean(),
            df["battery_voltage"].mean(),
            df["battery_voltage"].diff().mean(),
            df["relative_date_PCIAT"].tail(1).values[0]
        ]
        
        # List of columns of interest and masks
        keys = ["enmo", "anglez", "light", "enmo_weekend", "anglez_weekend"]
        masks = [no_mask, night, day, last_week]
        
        # Helper function for feature extraction
        def extract_stats(data):
            return [
                data.mean(), 
                data.std(), 
                data.max(), 
                data.min(), 
                data.kurtosis(), 
                data.skew(), 
                data.diff().mean(), 
                data.diff().std(), 
                data.diff().quantile(0.9), 
                data.diff().quantile(0.1)
            ]
        
        # Iterate over keys and masks to generate the statistics
        for key in keys:
            for mask in masks:
                filtered_data = df.loc[mask, key]
                features.extend(extract_stats(filtered_data))

        return features
    
    def process_file(self, filepath):
        ''' 
        This function takes the filepath of a time-series parquet file and extracts the id number and the
        statistics defined in extract_timeseries_features() function.
        '''
        df = pd.read_parquet(filepath)
        df.drop('step', axis=1, inplace=True)

        features = self.extract_timeseries_features(df)
        id_num = filepath.split('=')[-1]

        return features, id_num
    
    def timeseries_features_df(self, n_jobs=8):
        ''' 
        This function takes as input a filepath array and returns a dataframe with all the time-series statistics for each parquet file.
        The extraction process is sped up using Python's joblib library, with n_jobs being an optional argument.
        '''
        print('Extracting features from time-series files.')
        results = Parallel(n_jobs=n_jobs)(delayed(self.process_file)(f) for f in tqdm(self.filepath_array))

        stats, indices = zip(*results)
        df = pd.DataFrame(stats, columns=[f'stat_{i}' for i in range(len(stats[0]))])
        df['id'] = indices

        return df
    
if __name__ == '__main__':

    folder_path = './child-mind-institute-problematic-internet-use/'
    timeseries_train_files = glob.glob(folder_path + 'series_train.parquet/*')

    ts = time_series_features(timeseries_train_files)

    ts_stats = ts.timeseries_features_df()
    print(ts_stats)