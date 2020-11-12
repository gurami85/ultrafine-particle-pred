import pandas as pd

# read the data
df = pd.read_csv('data/dataset_jongno.csv', parse_dates=[2], date_parser=parser,
                       dtype={'index': int, 'station_id': int, 'normal_cnt': int, 'rapid_cnt': int})