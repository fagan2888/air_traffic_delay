
import glob
import pandas as pd
import matplotlib.pyplot as plt


# raw_traffic_path = 'data/air_traffic/*.zip'
# airports_data_path = 'data/airports.csv'

# read traffic data
def combine_traffic_data(files_dir, n):

    traffic_zip_files = glob.glob(files_dir)

    frames = []
    for z in traffic_zip_files[:n]: ### n to control how many to read
        print(z)
        temp_df = pd.read_csv(z, compression='infer').dropna(axis='columns', how='all')
        frames.append(temp_df)

    traffic = pd.concat(frames)

    del frames  # release some memeory

    return traffic


# add latitude/longitude to the traffic dataframe
def add_lat_long_to_traffic_data(traffic):

    airports_data_path = '../data/airports.csv' # for lat/long information

    airports = pd.read_csv(airports_data_path, compression='infer').dropna(axis='columns', how='all')

    def combine_columns(df1, left_on, prefix):

        df2 = airports[['AIRPORT_ID', 'LATITUDE', 'LONGITUDE']]
        right_on = ['AIRPORT_ID']

        df = df1.merge(df2.drop_duplicates(subset=['AIRPORT_ID']), left_on=left_on, right_on=right_on)

        df.rename(columns = {'LATITUDE':'{}LATITUDE'.format(prefix), 
                             'LONGITUDE': '{}LONGITUDE'.format(prefix)}, 
                             inplace = True)
        df.drop('AIRPORT_ID', axis=1, inplace=True)
        return df

    # lat/long of ORIGIN airports
    traffic = combine_columns(traffic, 
                             left_on=['ORIGIN_AIRPORT_ID'],
                             prefix='ORIGIN_')

    # lat/long of DESTINATION airports
    traffic = combine_columns(traffic,
                             left_on=['DEST_AIRPORT_ID'],
                             prefix='DEST_')

    return traffic



def add_labels(df, binary=True, DELAY_THRESHOLD=20, categorical=False):
    """add target labels and balance the data"""

    def delay_class(minutes):
        if minutes <= 5:
            return 0
        if 5 < minutes  <= 20:
            return 1
        if 20 < minutes <= 60:
            return 2
        if 60 < minutes <= 120:
            return 3
        if 120 < minutes:
            return 4
        else:
            return None

    if binary and not categorical:
        # add the target label "binary: delayed (positive) not-delayed (negative)" based on the threshold in minutes
        df['DELAYED'] = df['DEP_DELAY'].apply(lambda x: 1 if x >= DELAY_THRESHOLD else 0)

        # balance the data (same number of samples for delayed / not delayed flights)
        delayed = df[df['DELAYED'] == 1].copy()
        no_delay = df[df['DELAYED'] == 0][:delayed.shape[0]].copy()

        # concat into one dateframe
        data = delayed.append(no_delay, ignore_index=True)
        # logging
        percentage = delayed_percentage(df, DELAY_THRESHOLD)
        print('{:.2f}% of the total flights were delayed {} minutes or more.'.format(percentage, DELAY_THRESHOLD))

        del delayed, no_delay, df # release some memory

    elif categorical:
        df['DELAY_CLASS'] = df['DEP_DELAY'].apply(lambda row: delay_class(row))
        counts = df['DELAY_CLASS'].value_counts()
        m = min(counts)
        c0 = df[df['DELAY_CLASS'] == 0][:m].copy()
        c1 = df[df['DELAY_CLASS'] == 1][:m].copy()
        c2 = df[df['DELAY_CLASS'] == 2][:m].copy()
        c3 = df[df['DELAY_CLASS'] == 3][:m].copy()
        c4 = df[df['DELAY_CLASS'] == 4][:m].copy()
        data = c0.append([c1, c2, c3, c4])
        data['DELAY_CLASS'] = data['DELAY_CLASS'].astype(int)
        del c0, c1, c2, c3, c4 # release memory
    else:
        raise('either of binary or categorical must be true')

    # shuffle dataframe
    data = data.sample(frac=1).reset_index(drop=True)

    return data




def delayed_percentage(df, minute):
    """return the percentage of fligths that were delayed `minute` minutes"""
    frac = df[df['DEP_DELAY'] >= minute].shape[0] / df.shape[0]
    return frac * 100

