import pandas as pd
from datetime import datetime


# parser for date columns [date, start_time, end_time]
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


# read the data
df = pd.read_csv('data/dataset_jongno.csv', parse_dates=[0], date_parser=parser)

# reindex the data frame
column_names = ['datetime', 'month', 'traffic', 'temperature', 'humidity', 'dew_point_temperature'\
                , 'ground_surface_temperature', 'vapor_pressure', 'atmospheric_pressure'\
                , 'sea_level_pressure', 'sunshine', 'solar_radiation', 'precipitation'\
                , 'snowfall', 'cloud_cover', 'visibility', 'wind_speed', 'wind_direction'\
                , 'pm10']

df = df.reindex(columns=column_names)

# transformer for wind directions (from degree to categorical value)
def make_wind_direction(_df: pd.DataFrame) -> list:
    _wind_dir_lst = list()
    for idx, val in _df.iterrows():
        wind_dir_deg = val['wind_direction']
        if (wind_dir_deg >= 348.75) or (wind_dir_deg < 11.25):
            _wind_dir_lst.append('N')
        elif (wind_dir_deg >= 11.25) and (wind_dir_deg < 33.75):
            _wind_dir_lst.append('NNE')
        elif (wind_dir_deg >= 33.75) and (wind_dir_deg < 56.25):
            _wind_dir_lst.append('NE')
        elif (wind_dir_deg >= 56.25) and (wind_dir_deg < 78.75):
            _wind_dir_lst.append('ENE')
        elif (wind_dir_deg >= 78.75) and (wind_dir_deg < 101.25):
            _wind_dir_lst.append('E')
        elif (wind_dir_deg >= 101.25) and (wind_dir_deg < 123.75):
            _wind_dir_lst.append('ESE')
        elif (wind_dir_deg >= 123.75) and (wind_dir_deg < 146.25):
            _wind_dir_lst.append('SE')
        elif (wind_dir_deg >= 146.25) and (wind_dir_deg < 168.75):
            _wind_dir_lst.append('SSE')
        elif (wind_dir_deg >= 168.75) and (wind_dir_deg < 191.25):
            _wind_dir_lst.append('S')
        elif (wind_dir_deg >= 191.25) and (wind_dir_deg < 213.75):
            _wind_dir_lst.append('SSW')
        elif (wind_dir_deg >= 213.75) and (wind_dir_deg < 236.25):
            _wind_dir_lst.append('SW')
        elif (wind_dir_deg >= 236.25) and (wind_dir_deg < 258.75):
            _wind_dir_lst.append('WSW')
        elif (wind_dir_deg >= 258.75) and (wind_dir_deg < 281.25):
            _wind_dir_lst.append('W')
        elif (wind_dir_deg >= 281.25) and (wind_dir_deg < 303.75):
            _wind_dir_lst.append('WNW')
        elif (wind_dir_deg >= 303.75) and (wind_dir_deg < 326.25):
            _wind_dir_lst.append('NW')
        elif (wind_dir_deg >= 326.25) and (wind_dir_deg < 348.75):
            _wind_dir_lst.append('NNW')
    return _wind_dir_lst


wind_dir_lst = make_wind_direction(df)
df['wind_direction'] = wind_dir_lst

# save the refined data frame
df.to_csv('./data/dataset_jongno_refined.csv', index=False)