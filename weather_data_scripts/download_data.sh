stationID="4841"
output_file="climate_data/hourly/fixed_${stationID}.csv"
month="1"

> "$output_file"

for year in `seq 2007 2024`; do
    # for month in `seq 1 12`; do
        temp_file="climate_data/hourly/temp.csv"

        wget --server-response --no-check-certificate -P climate_data/hourly \
         "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=${stationID}&Year=${year}&Month=${month}&Day=14&timeframe=1&submit=Download+Data" \
         -O "$temp_file"

        if [[ $(wc -c < "$output_file") -eq 0 ]]; then
            cat "$temp_file" >> "$output_file"
        else
            tail -n +2 "$temp_file" >> "$output_file"
        fi

        rm "$temp_file"
    # done
done