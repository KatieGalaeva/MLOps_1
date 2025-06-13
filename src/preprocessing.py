import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from math import atan2, cos, radians, sin, sqrt
import logging

# Import extra modules
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer 

logger = logging.getLogger(__name__)
RANDOM_STATE = 42



def cat_encode(train, input_df, col):
    
    logger.debug('Encoding category: %s', col)
    new_col = col + '_cat'
    mapping = train[[col, new_col]].drop_duplicates()
    
    # Merge to initial dataset
    input_df = input_df.merge(mapping, how='left', on=col).drop(columns=col)
    
    return input_df


def add_distance_features(df):
    
    logger.debug('Calculating distances...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']), 
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df

def add_time_features(df):
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['year'] = df['transaction_time'].dt.year
    df['month'] = df['transaction_time'].dt.month
    df['day'] = df['transaction_time'].dt.day
    df['hour'] = df['transaction_time'].dt.hour
    df['minute'] = df['transaction_time'].dt.minute
    df['day_of_week'] = df['transaction_time'].dt.dayofweek
    return df.drop(columns=['transaction_time'])

def add_geo_features(df):
    df['bearing_degree_1'] = df.apply(lambda x: bearing_degree(
        x['lat'], x['lon'], x['merchant_lat'], x['merchant_lon']), axis=1)
    df['hav_dist_1'] = df.apply(lambda x: haversine_distance(
        x['lat'], x['lon'], x['merchant_lat'], x['merchant_lon']), axis=1)
    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, n_digits: int = 0) -> float:
    """
    Function for calculating distances from points A to B in a straight line

        :param lat1: Latitude of point A
        :param lon1: Longitude of point A
        :param lat2: Latitude of point B
        :param lon2: Longitude of point B
        :param n_digits: Round the answer to n decimal places
        :return: Distance straight from the edge to n_digits
    """

    lat1, lon1, lat2, lon2 = round(lat1, 6), round(lon1, 6), round(lat2, 6), round(lon2, 6)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)

    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    return round(2 * 6372800 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), n_digits)  


def bearing_degree(lat1: float, lon1: float, lat2: float, lon2: float, n_digits: int = 0) -> float:
    """
    Function for calculating the angle between the line [((lat1, lon1), (lat2, lon2)), (zero meridian)]

        :param lat1: Latitude of point A
        :param lon1: Longitude of point A
        :param lat2: Latitude of point B
        :param lon2: Longitude of point B
        :param n_digits: Round the answer to n decimal places
        :return: Angle value with an accuracy of n_digits
    """

    lat1, lon1 = np.radians(round(lat1, 6)), np.radians(round(lon1, 6))
    lat2, lon2 = np.radians(round(lat2, 6)), np.radians(round(lon2, 6))

    dlon = (lon2 - lon1)
    numerator = np.sin(dlon) * np.cos(lat2)
    denominator = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

    theta = np.arctan2(numerator, denominator)
    theta_deg = (np.degrees(theta) + 360) % 360

    return round(theta_deg, n_digits)

# Calculate means for encoding at docker container start
def load_train_data():

    logger.info('Loading training data...')

    # Define column types
    target_col = 'target'
    categorical_cols = ['merch', 'cat_id', 'name_1', 'name_2', 'gender', 'street', 'one_city', 'us_state', 'jobs', 'year',	'month', 'day',	'hour',	'minute','day_of_week']
    n_cats = 50

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some simple time features
    train = add_time_features(train)
    train = add_geo_features(train)

    for col in categorical_cols:
        new_col = col + '_cat'

        # Get table of categories
        temp_df = train\
            .groupby(col, dropna=False)[[target_col]]\
            .count()\
            .sort_values(target_col, ascending=False)\
            .reset_index()\
            .set_axis([col, 'count'], axis=1)\
            .reset_index()
        temp_df['index'] = temp_df.apply(lambda x: np.nan if pd.isna(x[col]) else x['index'], axis=1)
        temp_df[new_col] = ['cat_NAN' if pd.isna(x) else 'cat_' + str(x) if x < n_cats else f'cat_{n_cats}+' for x in temp_df['index']]

        train = train.merge(temp_df[[col, new_col]], how='left', on=col)
    

    logger.info('Train data processed. Shape: %s', train.shape)

    return train


# Main preprocessing function
def run_preproc(train, input_df, model_features, numeric_feat):
    logger.info('Running preprocessing pipeline...')
    
    df = input_df.copy()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['ts_transaction_time'] = df['transaction_time'].astype(np.int64) // 10**9  # в секундах

    df = add_time_features(df)
    df = add_geo_features(df)

    df = df[[col for col in model_features if col in df.columns]]

    logger.info('Preprocessing completed. Final shape: %s', df.shape)
    
    return df

