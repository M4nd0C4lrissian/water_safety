import pandas as pd
from datetime import datetime
from datetime import timedelta

# assumes data in date-order
def detect_gaps(data, dt_format = "%Y-%m-%d"):
    
    data = data['CollectionDate']
    gaps = []
    
    for i in range(0, data.shape[0]-1):
        row = data.iloc[i]
        cur_dt = datetime.strptime(row, dt_format)
        
        inc_dt = cur_dt + timedelta(days=1)
        
        next_row = data.iloc[i+1]
        next_dt = datetime.strptime(next_row, dt_format)
        
        if next_dt.date() == inc_dt.date():
            continue 
        elif next_dt.strftime("%Y") == cur_dt.strftime("%Y"):
            gaps.append((cur_dt, next_dt))
            
    return gaps

def fill_gaps(df, gaps, dt_format="%Y-%m-%d"):
    filled_rows = []

    for start, end in gaps:
        delta_days = (end - start).days - 1
        if delta_days <= 0:
            continue

        x0 = df[df['CollectionDate'] == start.strftime(dt_format)].iloc[0]
        x1 = df[df['CollectionDate'] == end.strftime(dt_format)].iloc[0]
        beach_id = x0['beachId']
        e0 = x0['eColi']
        e1 = x1['eColi']

        for i in range(1, delta_days + 1):
            new_date = start + timedelta(days=i)
            new_ecoli = e0 + i * (e1 - e0) / (delta_days + 1)
            filled_rows.append({
                'CollectionDate': new_date.strftime(dt_format),
                'beachId': beach_id,
                'eColi': round(new_ecoli),
                'statusFlag': 0
            })

    if filled_rows:
        filled_df = pd.DataFrame(filled_rows)
        df = pd.concat([df, filled_df], ignore_index=True)
        df['CollectionDate'] = pd.to_datetime(df['CollectionDate'])
        df = df.sort_values(by='CollectionDate').reset_index(drop=True)
        df['CollectionDate'] = df['CollectionDate'].dt.strftime(dt_format)

    return df.reset_index(drop=True)

# they seem to miss gaps of 1 to 2 days a few times a year - what do we do? - just not include?
if __name__ == '__main__':
    
    beaches = ['HanlansPoint', 'GibraltarPoint', 'CherryBeach', 'WardsIsland', 'CentreIslandBeach']
    
    for beach_name in beaches:
        
        data = pd.read_csv(f'water_safety\ecoli_readings\cleaned_{beach_name}.csv', index_col=0)
        gaps = detect_gaps(data)
        
        print('Gaps in data - [ start_date, end_date ]')
        for gap in gaps:
            print(f'{gap[0].strftime("%Y-%m-%d")}, {gap[1].strftime("%Y-%m-%d")} : total diff - {(gap[1] - gap[0]).days}')
            
        df_filled = fill_gaps(data, gaps)

        # Optionally save the updated dataframe
        df_filled.to_csv(f'water_safety\ecoli_readings\\filled_{beach_name}.csv')
        
        gaps = detect_gaps(df_filled)
        
        print('Gaps in data - [ start_date, end_date ]')
        for gap in gaps:
            print(f'{gap[0].strftime("%Y-%m-%d")}, {gap[1].strftime("%Y-%m-%d")} : total diff - {(gap[1] - gap[0]).days}')
            
        
# 2009 hanlan's point should be ignored
        
        