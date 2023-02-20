import pandas as pd
import numpy as np
from skimpy import clean_columns
import os
import re 


def clean_hxh_table(season):
    url = "https://en.wikipedia.org/wiki/Hunter_%C3%97_Hunter_(2011_TV_series)"

    df_raw = pd.read_html(url)[season]\
        .pipe(clean_columns)\
        .filter(['no_overall','no_in_season','title','original_air_date'])

    df = pd.concat(
        [df_raw[::2]\
            .reset_index(drop=True),
        (df_raw[1::2]\
            .filter(['title'])\
            .rename(columns={"title": "description"})\
            .reset_index(drop=True))],
        axis=1)
    
    df = df.assign(original_air_date = lambda x: 
              x.original_air_date.str.replace(r"\s*\[\d+\]\s*", ""))\
           .assign(Episode = lambda x: x['title'].str.split('"', 2).str[1])
    
    df = df.assign(
        no_overall = lambda x: x.no_overall.astype(int),
        no_in_season = lambda x: x.no_in_season.astype(int),
        original_air_date = lambda x: pd.to_datetime(x.original_air_date, format='%B %d, %Y'),
    )
    
    return df

hxh_list = []

for i in range(6):
    hxh_list.append(clean_hxh_table(i+1))

hxh_episodes = pd.concat(hxh_list)


#%% 

hxh_subs = np.array([])

for i in range(1, 149):
    file_path = f"hxhsubs/Hunter x Hunter - {i}.enUS.csv"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            hxh_subs = np.append(hxh_subs, f.read())

hxh_subs = np.vectorize(lambda x: re.sub(r"\n\d+;\d+;\d+;", "", x))(hxh_subs)

hxh_episodes = hxh_episodes.sort_values(by=['original_air_date']).reset_index(drop=True)
hxh_episodes['hxh_subs'] = pd.Series([word[68:] for word in list(hxh_subs)])

hxh_episodes.to_csv("hxh_episodes.csv", index=False)