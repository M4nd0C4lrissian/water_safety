import pandas as pd
from datetime import datetime
from datetime import timedelta

# assumes data in date-order
def detect_gaps(data, dt_format = "%Y-%m-%d"):
    
    data = data['Date/Time']
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

# they seem to miss gaps of 1 to 2 days a few times a year - what do we do? - just not include?
if __name__ == '__main__':
    data = pd.read_csv('water_safety\weather_data_scripts\cleaned_data\daily\cleaned_merged_toronto_city_hanlans.csv', index_col=0)
    gaps = detect_gaps(data)
    
    print('Gaps in data - [ start_date, end_date )')
    for gap in gaps:
        print(f'{gap[0].strftime("%Y-%m-%d")}, {gap[1].strftime("%Y-%m-%d")}')
        
        
        
        