import pandas as pd

## count null

def count_null(path):
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    count = df.isna().sum()
    print(count)
    return count

def count_non_numeric(path, col):
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    mask = pd.to_numeric(df[col], errors='coerce').isna()
    count = mask.sum()

    print(count)
    return count

## remove bad data

def remove_val_from_col(path, out_path, col, val):
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    mask = (df[col] == val)

    df = df.drop(df.loc[mask].index)

    # reverse df to be in increasing date order
    df = df.iloc[::-1]

    df.index = range(df.shape[0])
    df.to_csv(out_path)

## detect and log gaps in readings

def find_continuous_dates():
    pass


if __name__ == '__main__':

    # count_null('water_safety\ecoli_readings\HanlansPoint.csv')

    # no null values

    # remove non-numeric
    # count_non_numeric('water_safety\ecoli_readings\HanlansPoint.csv', 'statusFlag')

    remove_val_from_col('water_safety\ecoli_readings\HanlansPoint.csv',
    'water_safety\ecoli_readings\cleaned_HanlansPoint.csv', 'statusFlag', 'UNTESTED')

    # count_non_numeric('water_safety\ecoli_readings\cleaned_HanlansPoint.csv', 'statusFlag')

