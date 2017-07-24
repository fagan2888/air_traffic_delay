

import pandas as pd
import numpy as np
import os.path
import catboost
from sklearn import preprocessing
from preprocess_data import combine_traffic_data, add_lat_long_to_traffic_data, add_labels


def us_air_traffic_data(data_folder=None, label_type='binary', DELAY_THRESHOLD=20, n=14):

    if not data_folder:
        data_folder = '../data/air_traffic/*.zip'
    
    p('reading air traffic data ... ','')
    raw_air_traffic = combine_traffic_data(data_folder, n)
    p('\tdata shape:', raw_air_traffic.shape)

    p('adding lat/long info ...','')
    air_traffic = add_lat_long_to_traffic_data(raw_air_traffic)
    p('\tdata shape:', air_traffic.shape)
    
    p('preparing labels ...','')
    category = True if label_type == 'multiclass' else False
    data = add_labels(air_traffic, categorical=category)
    p('\nfinal prepared data shape:', data.shape)
    return data


def p(s, v):
    print('{} {}'.format(s, v))

def transofrm_data(data=None, binary=True, categorical=False):

    if not isinstance(data, pd.DataFrame):
        raise('no DataFrame!')


    print('normalizing and preparing training data ...')
    
    if binary and not categorical:
        target = data['DELAYED']  # the label
        data.drop('DELAYED', axis=1, inplace=True)
    elif categorical:
        target = data['DELAY_CLASS']
        data.drop('DELAY_CLASS', axis=1, inplace=True)

    # drop the target label from the features, and 
    # the delay-indicator columns e.g. DEP_DELAY, ARR_DELAY ... etc (very important!)
    to_drop = data.columns[data.columns.str.contains('DEL')].values
    data.drop(to_drop, axis=1, inplace=True)

    # try out some magic
    data = extract_features(data)


    # non numerical column indecies
    categorical_idx = [data.columns.get_loc(c) for c in data.columns[data.dtypes == 'object']]

    pool = catboost.Pool(data.fillna(0), target,
                         has_header=True, 
                         cat_features=categorical_idx)

    X = pool.get_features()
    y = pool.get_label()

    
    # normalize
    X, y = np.array(X).astype(float), np.array(y).astype(float)
    X[np.isnan(X)] = 0
    X = preprocessing.normalize(X)

    del pool  # release memory

    # TODO: feature mapping before training

    return X, y, data



def extract_features(df):
    # TODO: better features
    
    # drop duplicate and training-unnecessary columns
    cols_to_drop = [
    'FL_DATE',

    'AIRLINE_ID',
    'ORIGIN_AIRPORT_ID', 'ORIGIN_CITY_NAME',
    'DEST_AIRPORT_ID', 'DEST_CITY_NAME',

    # to reduce the complexity of the model
    'FL_NUM',
    'DEP_TIME', # instead DEP_TIME_BLK
    'ARR_TIME', # instead ARR_TIME_BLK
    'CRS_DEP_TIME',
    'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME',
    'WHEELS_OFF',
    'WHEELS_ON',
    'TAIL_NUM',
    'ACTUAL_ELAPSED_TIME',
    ]

    df.drop(cols_to_drop, axis=1, inplace=True)

    
    # grouping values (rounding to nearest nth)
    def round_to(col, n):
        return df[col].apply(lambda x: round(x,n))

    df['DISTANCE'] = round_to('DISTANCE', -3) # about 6 groups
    df['TAXI_IN'] = round_to('TAXI_IN', -1)  # about 14 groups
    df['TAXI_OUT'] = round_to('TAXI_OUT', -1)  # about 14 groups
    df['DEST_LATITUDE'] = round_to('DEST_LATITUDE', -1)
    df['DEST_LONGITUDE']= round_to('DEST_LONGITUDE', -1)
    df['ORIGIN_LATITUDE'] = round_to('ORIGIN_LATITUDE' ,-1)
    df['ORIGIN_LONGITUDE']= round_to('ORIGIN_LONGITUDE' ,-1)
    df['AIR_TIME'] = round_to('AIR_TIME', -3)


    return df



