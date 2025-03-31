import numpy as np
import pandas as pd

def merge_duplicates(data_path, out_path):
    df = pd.read_csv(data_path, index_col=0)
    
    df = df.groupby('Date/Time').first().reset_index()
    
    df.to_csv(out_path)
    
def combine_weather_water(water_data, weather_data, out_path):

    new_df = water_data.merge(weather_data, how='inner', left_on='CollectionDate', right_on='Date/Time')
    new_df.to_csv(out_path)

def find_viable_columns(data_path, out_path, rel_tolerance = 1):

    df = pd.read_csv(data_path, index_col=0)

    empty_cols = []
    for col in df.columns:
        if df[col].count() < df.shape[0]* (1 - rel_tolerance) or len(df[col].dropna().unique()) <= 1:
            empty_cols.append(col)

    df.drop(columns=empty_cols, inplace=True)
    # df.drop(columns=['Longitude (x)', 'Latitude (y)', 'Station Name', 'Climate ID'], inplace=True)

    print(df.columns)
    print('-----------------------')

    # for c in df.columns:
    #     print(df[c].describe())
    print(df.describe())


    print('--------------------------------------')

    df.to_csv(out_path)

# mostly empty: Max Temp (°C),  Min Temp (°C),  Mean Temp (°C),  Heat Deg Days (°C), Cool Deg Days (°C), Cool Deg Days Flag
# Total rain only about 50000 out of 80000 total (it says only 1 unique value)

# most of the wind and snow stuff
# a lot of values are just M

if __name__ == '__main__':
    # merge duplicates 
    data_path = 'water_safety\weather_data_scripts\climate_data\daily\\31688.csv'    
    merge_duplicates(data_path, data_path)
    # combine weather and water readings
    water_data = pd.read_csv('water_safety\ecoli_readings\cleaned_HanlansPoint.csv', index_col=0)
    weather_data = pd.read_csv(data_path, index_col=0).drop(columns=['Unnamed: 0.1', 'Unnamed: 0.2', 'Unnamed: 0'])
    merge_path = 'water_safety\weather_data_scripts\cleaned_data\daily\\merged_toronto_hanlans.csv'
    combine_weather_water(water_data, weather_data, merge_path)
    # filter out bad_data
    out_path = 'water_safety\weather_data_scripts\cleaned_data\daily\\merged_toronto_hanlans.csv'
    find_viable_columns(merge_path, out_path, rel_tolerance=1)