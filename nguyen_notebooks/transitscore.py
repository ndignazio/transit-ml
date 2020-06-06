import pandas as pd
from pandas.io.json import json_normalize
import requests 


def get_data(lat, lng):
    '''
    Gets data for one location.
    
    Inputs:
        lat, lng (str)
    Returns:
        (json)
    '''
    rv = None
    key = '4c4eb18a1eea25128110eaf683aefab4'
    url = 'https://transit.walkscore.com/transit/score/?lat={}&lon={} \
    	   &wsapikey={}'.format(lat, lng, key)
    r = requests.get(url)
    if not r.status_code==400:
        rv = r.json()

    return rv


def create_datalist(df):
    '''
    Turns list of jsons into a dataframe
    
    Inputs:
        acsse data frame (pandas Dataframe)
    Returns:
        list containing transit score for all places for which it 
        is available (list of json)
    '''
    datalist = []
    
    for row in df.itertuples():
        data = get_data(row.Lat, row.Lon)
        if not data is None:
            data["city"] = row.city
            datalist.append(data)
    
    return datalist


def create_dataframe(data_list):
    '''
    Given the list of json objects containing transit score data, 
    create dataframe.

    Inputs:
    	(list of json)
    Returns:
    	(pandas DataFrame)
    '''    
    df = pd.DataFrame.from_dict(json_normalize(data_list), orient='columns')
    df = df.drop(columns=['help_link', 'logo_url'])
    df[['nearby_routes', 'bus', 'rail', 'other']] = df['summary'].str\
    										.findall(r'\d+').apply(pd.Series)
    
    return df

