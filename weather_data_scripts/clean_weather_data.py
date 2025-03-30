import numpy as np
import pandas as pd


def find_viable_columns(data_path, out_path, rel_tolerance = 1):

    df = pd.read_csv(data_path)

    empty_cols = []
    for col in df.columns:
        if df[col].count() < df.shape[0] - rel_tolerance * df.shape[0] or len(df[col].dropna().unique()) <= 1:
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
    data_path = 'water_safety\weather_data_scripts\climate_data\hourly\\31688.csv'
    out_path = 'water_safety\weather_data_scripts\cleaned_data\hourly\\toronto_city.csv'

    find_viable_columns(data_path, out_path, rel_tolerance=1)