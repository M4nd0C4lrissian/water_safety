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

if __name__ == '__main__':
    
    beach_name = 'CentreIslandBeach'

    count_null(f'water_safety\ecoli_readings\\{beach_name}.csv')

    # no null values

    # remove non-numeric
    count_non_numeric(f'water_safety\ecoli_readings\\{beach_name}.csv', 'statusFlag')

    remove_val_from_col(f'water_safety\ecoli_readings\\{beach_name}.csv',
    f'water_safety\ecoli_readings\\cleaned_{beach_name}.csv', 'statusFlag', 'UNTESTED')

    count_non_numeric(f'water_safety\ecoli_readings\\cleaned_{beach_name}.csv', 'statusFlag')
    pass
