import json
import pandas as pd
from datetime import datetime

## new format - Year, Month, Day, beachId, eColi, statusFlag (binary)

## check if data is null, then if eColi is null

## extract CollectionDate + other qualities listed above
## change CollectionDate into three values and make a list of dicts
## also have flags to set categoricals into binary (statusFlag)
## remove null values

def beach_to_csv(json_path, out_path, date_name = 'CollectionDate', cols = ['beachId', 'eColi'], split_date = True, categorical_cols = [('statusFlag', 'UNSAFE', 'SAFE')]):

    with open(json_path) as file:
        d = json.load(file)
    
    new_entries = []
    for i in range(len(d)):
        entry = d[i]

        if entry['data'] is None:
            continue

        data = entry['data'][0]

        if data['eColi'] is None:
            continue

        new_dict = {}

        if split_date:
            date_str = entry[date_name]

            dt = datetime.strptime(date_str, "%Y-%m-%d")
            year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
            new_dict['Year'] = year
            new_dict['Month'] = month
            new_dict['Day'] = day
        else:
            new_dict[date_name] = entry[date_name]

        for col in cols:
            new_dict[col] = data[col]

        for cat in categorical_cols:
            name = cat[0]
            high_cat = cat[1]
            low_cat = cat[2]

            if data[name] == high_cat:
                new_dict[name] = 1
            elif data[name] == low_cat:
                new_dict[name] = 0
            else:
                new_dict[name] = data[name]

        new_entries.append(new_dict)

    cat_cols = [c[0] for c in categorical_cols]
    final_cols = cols + cat_cols

    if split_date:
        final_cols = ['Year', 'Month', 'Day'] + final_cols
    else:
        final_cols = [date_name] + final_cols

    df = pd.DataFrame(new_entries, columns= final_cols)
    df.to_csv(out_path)
            

if __name__ == '__main__':
    
    beach_name = 'CentreIslandBeach'
    
    beach_to_csv(f'water_safety\ecoli_readings\\beach_data_raw\\{beach_name}.json', f'water_safety\ecoli_readings\\{beach_name}.csv', split_date = False)


## NOTE -------
# After this, there are still a few rows that have non null entries but have other strings for
# statusFlag - we have to be wary of this kind of thing - I'll clean it next