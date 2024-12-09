import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Union, Tuple   
from scipy import stats
import numpy as np
import pandas as pd

class time_series_features:
    ''' 
    This class contains methods for extracting statistics from the actigraphy parquet files. The class can be instantiated with only one argument
    where 'file_path_array' is an array containing the file paths of all the parquet files.

    If 'extract_timeseries_features()' and/or 'process_file()' methods are to be used standalone, then an extra parameter is required.
    '''
    def __init__(self, filepath_array : List[str]) -> None:
        self.filepath_array = filepath_array

    def extract_timeseries_features(self, df : pd.DataFrame) -> List[Union[int, float]]:
        ''' 
        This function takes in as argument actigraphy timeseries dataframe and returns a list of useful features to be merged with
        the CMI dataframe.  
        '''
        df = df.copy()
        
        # Creating timestamps from 'relative_date_PCIAT' and 'time_of_day'
        df['timestamp'] = pd.to_datetime(df['relative_date_PCIAT'], unit='D') + pd.to_timedelta(df['time_of_day']) 
        df = df[df['non-wear_flag'] == 0]
        
        # Calculating basic metrics
        df['magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        df['velocity'] = df['magnitude']
        df['distance'] = df['velocity'] * 5 # 5 seconds per observation
        df['date'] = df['timestamp'].dt.date
        df['hour'] = pd.to_datetime(df['time_of_day']).dt.hour
        
        # Calculating aggregated distances using using 'timestamp' and 'date' characterizing daily and monthly movement
        distances = {
            'daily'   : df.groupby('date')['distance'].sum(),
            'monthly' : df.groupby(df['timestamp'].dt.to_period('M'))['distance'].sum()
        }
        
        # Features dictionary initialization - here the different extracted features will be stored
        features = {}
        
        # Time masks demarcating different periods of the day
        # morning : between 6 am and 12 pm, afternoon : between noon and 6 pm, evening : between 6 pm and 10 pm, night : between 10 pm and 6 am
        time_masks = {
            'morning'  : (df['hour'] >= 6) & (df['hour'] < 12),
            'afternoon': (df['hour'] >= 12) & (df['hour'] < 18),
            'evening'  : (df['hour'] >= 18) & (df['hour'] < 22),
            'night'    : (df['hour'] >= 22) | (df['hour'] < 6)
        }
        
        # Activity patterns: these are measured as movement during specified daily time periods
        for period, mask in time_masks.items():
            features.update(
                {
                    f'{period}_activity_mean' : df.loc[mask, 'magnitude'].mean(),
                    f'{period}_activity_std'  : df.loc[mask, 'magnitude'].std(),
                    f'{period}_enmo_mean'	  : df.loc[mask, 'enmo'].mean(),
                    f'{period}_enmo_std'	  : df.loc[mask, 'enmo'].std()	
                }
            )
            
        # Sleep quality: this is measured as movement during night time using a defined threshold
        # sleep_disruption_count : counts no. of occurrences of movement beyond defined threshold
        # sleep_position_changes : counts no. times wrist position changes by over 45 degrees
        sleep_hours = time_masks['night']
        magnitude_threshold = df['magnitude'].mean() + df['magnitude'].std()
        
        features.update(
            {
                'sleep_movement_mean'         : df.loc[sleep_hours, 'magnitude'].mean(),
                'sleep_movement_std'          : df.loc[sleep_hours, 'magnitude'].std(),
                'sleep_disruption_count'      : len(
                    df.loc[
                        sleep_hours & (df['magnitude'] > magnitude_threshold)
                    ]
                ),
                'light_exposure_during_sleep' : df.loc[sleep_hours, 'light'].mean(),
                'sleep_position_changes'      : len(
                    df.loc[
                        sleep_hours & (abs(df['anglez'].diff() > 45))
                    ]
                ),
                'good_sleep_cycle'            : int(df.loc[sleep_hours, 'light'].mean() < 50)
            }
        )
        
        # Activity intensity: this is measures intensity of various activities from sedentary to vigorous using addition threshold combined with 'magnitude_threshold'
        features.update(
            {
                'sedentary_time_ratio'    : (df['magnitude'] < magnitude_threshold * 0.5).mean(),
                'moderate_activity_ratio' : (
                    (df['magnitude'] >= magnitude_threshold * 0.5) & (df['magnitude'] < magnitude_threshold * 1.5)
                ).mean(),
                'vigorous_activity_ratio' : (df['magnitude'] >= magnitude_threshold * 1.5).mean(),
                'activity_peaks_per_day'  : len(
                    df[
                        df['magnitude'] > df['magnitude'].quantile(0.95)
                    ]
                ) / len(df.groupby('relative_date_PCIAT')) 
            }
        )
        
        # Circadian rhythm: this measures the circadian rhythm's regularity  by analyzing the highs and lows of hourly activity
        hourly_activity = df.groupby('hour')['magnitude'].mean()
        features.update(
            {
                'circiadian_regularity' : hourly_activity.std() / hourly_activity.mean(),
                'peak_activity_hour'    : hourly_activity.idxmax(),
                'low_activity_hour'     : hourly_activity.idxmin(),
                'activity_range'        : hourly_activity.max() - hourly_activity.min()
                
            }
        )
        
        # Additional features: this filters out weekends for additional analysis
        weekend_mask = df['weekday'].isin([6, 7])
        
        features.update(
            {
                # movement patterns
                'movement_entropy'  		 : stats.entropy(
                    pd.qcut(df['magnitude'], q = 10, duplicates = 'drop').value_counts()
                ),
                'direction_changes' 		 : len( df[ abs(df['anglez'].diff()) > 30 ] ) / len(df),
                'sustained_activity_periods' : len( df[ df['magnitude'].rolling(12).mean() > magnitude_threshold ] ) / len(df),
                # weekend and weekday activities comparison
                'weekend_activity_ratio'     : df.loc[weekend_mask, 'magnitude'].mean() / df.loc[~weekend_mask, 'magnitude'].mean(),
                'weekend_sleep_difference'   : df.loc[weekend_mask & sleep_hours, 'magnitude'].mean() - df.loc[~weekend_mask & sleep_hours, 'magnitude'].mean(),
                # non-wear time
                'wear_time_ratio' 			 : (df['non-wear_flag'] == 0).mean(),
                'wear_consistency'			 : len(df['non-wear_flag'].value_counts()),
                #'longest_wear_streak'		 : df['non-wear_flag'].eq(0).astype(int).groupby(df['non-wear_flag'].ne(0).cumsum().sum().max()),
                # device usage
                'screen_time_proxy'      : ( df['light'] > df['light'].quantile(0.75).mean() ).sum() / len(df),
                'dark_environment_ratio' : ( df['light'] < df['light'].quantile(0.25).mean() ).sum() / len(df),
                'light_variation'		 : df['light'].std() / df['light'].mean() if df['light'].mean() != 0 else 0,	
                # battery usage
                'battery_drain_rate'	 : -np.polyfit( range(len(df)), df['battery_voltage'], 1 )[0],
                'battery_variability'	 : df['battery_voltage'].std(),
                'low_battery_time'		 : ( df['battery_voltage'] < df['battery_voltage'].quantile(0.1) ).mean(),
                # time-based 
                'days_monitored'		 : df['relative_date_PCIAT'].nunique(),
                'total_active_hours'	 : (len( df[ df['magnitude'] > magnitude_threshold * 0.5 ] ) * 5) / 3600,
                'activity_regularity'	 : df.groupby('weekday')['magnitude'].mean().std() 
            }
        )

        for col in ['X', 'Y', 'Z', 'enmo', 'anglez']:
            features.update(
                {
                    f'{col}_skewness' : df[col].skew(),
                    f'{col}_kurtosis' : df[col].kurtosis(),
                    f'{col}_trend'	  : np.polyfit(range(len(df)), df[col], 1)[0]
                }
            )

            return list(features.values())

    def process_file(self, filepath : str) -> Tuple[List[Union[int, float]], str]:
        ''' 
        This function takes the filepath of a time-series parquet file and extracts the id number and the
        statistics defined in extract_timeseries_features() function.
        '''
        df = pd.read_parquet(filepath)
        df.drop('step', axis=1, inplace=True)

        features = self.extract_timeseries_features(df)
        id_num = filepath.split('=')[-1]

        return features, id_num

    def extract_timeseries_id(filepath_array : List[str]) -> List[str]:
        ''' 
        This function takes an array of filepaths and returns an array of id numbers.
        '''
        id_list = []
        for n, folder in enumerate(filepath_array):
            id = folder.split('=')[-1]
            id_list.append(id)

        return id_list 

    def timeseries_features_df(self, filepath_array : List[str], n_jobs : int =8) -> pd.DataFrame:
        ''' 
        This function takes as input a filepath array and returns a dataframe with all the time-series statistics for each parquet file.
        The extraction process is sped up using Python's joblib library, with n_jobs being an optional argument.
        '''
        print('Extracting features from time-series files.')
        results = Parallel(n_jobs=n_jobs)(delayed(self.process_file)(f) for f in tqdm(filepath_array))

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