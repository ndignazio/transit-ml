import pandas as pd
import geopandas as gpd
from pandas.io.json import json_normalize
import requests 

tracts_filepath = 'shape_tracts/tl_2018_17_tract.shp'
places_filepath = 'shape_places/tl_2018_17_place.shp'
jobs_filepath = 'il_jobs_by_tract_2017.csv'


def get_data(lat, lng, city):
    '''
    Gets data for one location.
    
    Inputs:
        lat, lng (str)
    Returns:
        (json)
    '''
    rv = None
    over_quota = False
    
    #key = '4c4eb18a1eea25128110eaf683aefab4' #Nguyen original key
    #key = 'ffd1c56f9abcf84872116b4cc2dfcf31' #Mike key1
    #key = '4c4eb18a1eea25128110eaf683aefab4' #Mike key2
    #key = 'ffd1c56f9abcf84872116b4cc2dfcf31' #Nathan key1
    #key = '4c4eb18a1eea25128110eaf683aefab4' #Nathan key2
    key = 'ffd1c56f9abcf84872116b4cc2dfcf31' #NEW KEY
    
    url = 'https://transit.walkscore.com/transit/score/?lat={}&lon={}&city={}&state=IL&wsapikey={}'.format(lat, lng, city, key)
    
    r = requests.get(url)
    if not r.status_code==400:
        rv = r.json()
    if r.text == 'Over quota.':
        over_quota = True
        print('error: over quota')
    print(r)
    return rv, over_quota


def create_datalist(df):
    '''
    Turns list of jsons into a dataframe
    
    Inputs:
        acsse data frame (pandas Dataframe)
    Returns:
        list containing transit score for all places for which it is available (list of json)
    '''
    datalist = []
    over_quota_count = 0
    
    for row in df.itertuples():
        print(row.centroid_lat, row.centroid_lng, row.place_name)
        json, quota_bool = get_data(row.centroid_lat, row.centroid_lng, row.place_name)
        if not json is None:
            json['tract_GEO_ID'] = row.tract_GEO_ID
            json['centroid_lat'] = row.centroid_lat
            json['centroid_lng'] = row.centroid_lng
            datalist.append(json)
        if quota_bool:
            over_quota_count += 1
    return datalist, over_quota_count


def create_dataframe(datalist):
    '''
    Given the list of json objects containing transit score data, 
    create dataframe.

    Inputs:
    	(list of json)
    Returns:
    	(pandas DataFrame)
    '''    
    df = pd.DataFrame.from_dict(json_normalize(datalist), orient='columns')
    df = df.drop(columns=['help_link', 'logo_url'])
    df[['nearby_routes', 'bus', 'rail', 'other']] = df['summary'].str\
    										.findall(r'\d+').apply(pd.Series)
    return df


def tract_data(tracts_filepath, jobs_filepath):
    '''
    Loads and cleans census tract shapefiles.
    Creates a dataframe of all data on the tract level.

    Inputs:
        tracts_filepath, jobs_filepath (str)
    Outputs:
        (geopandas DataFrame)
    '''
    #Loading tracts
    tracts = gpd.read_file(tracts_path)
    tracts = tracts[['GEOID', 'NAMELSAD', 'ALAND', 'geometry']] \
                            .rename(columns={'GEOID': 'tract_GEO_ID', 'NAMELSAD': 'tract_name',
                           'ALAND': 'tract_area'})
    tracts['centroid_lng'] = tracts.geometry.centroid.x
    tracts['centroid_lat'] = tracts.geometry.centroid.y
    
    #Loading jobs
    jobs = pd.read_csv(jobs_filepath)
    jobs = jobs[['id', 'label', 'c000']] \
            .rename(columns={'id': 'job_tract_GEO_ID', 'label': 'job_tract_label',
                             'c000': 'num_jobs'})
    jobs['job_tract_GEO_ID'] = jobs['job_tract_GEO_ID'].astype(str)
    
    #Linking tracts and jobs
    df = pd.merge(jobs, tracts, how='inner', left_on='job_tract_GEO_ID', right_on='tract_GEO_ID')
    return df


def place_data(places_filepath):
    '''
    Loads and cleans places shapefiles.

    Inputs:
        places_filepath (str)
    Outputs:
        (geopandas DataFrame)
    '''
    places = gpd.read_file(places_path)
    places = places[['GEOID', 'NAME', 'NAMELSAD', 'geometry']] \
                            .rename(columns={'GEOID': 'place_GEO_ID', 'NAME': 'place_name',
                           'NAMELSAD': 'place_name_and_type'})
    return places


def get_transitscore(tracts, places):
    '''
    Spatially joins tracts and places.
    Gets transit score for census tracts.
    Merges tract data, place data, and transit score data.

    Inputs:
        tracts, places (geopandas DataFrame)
    Outputs:
        (pandas DataFrame)
    '''
    #Merging tracts and places
    tracts_places = gpd.sjoin(tracts, places, how="inner", op="intersects")

    #Getting transit score data
    datalist, over_quota_count = create_datalist(tracts_places)
    transitscore_data = create_dataframe(datalist)

    df = tracts_places.merge(transitscore_data, on='tract_GEO_ID')
    return df 


def data_cleaning(df):
    '''
    Given full dataframe, calculates job density and population density,
    drops rows with zero population, and changes NaN values to zero.
    '''
    df['job_density'] = df['num_jobs'] / ((df['tract_area'])/1000000)
    df['pop_density'] = df['race_total'] / ((df['tract_area'])/1000000)

    index_names = df[df['race_total']==0].index
    df.drop(index_names , inplace=True)

    df = df.fillna(0)
    return df



def go(acs5, tracts_filepath, places_filepath, jobs_filepath, pickle_filename):
    '''
    Executes all steps to take pulled ACS data to final dataframe for model 
    selection.

    Inputs:
        acs5 (pandas DataFrame)
        tracts_filepath, places_filepath, jobs_filepath, pickle_filename (str)
    Outputs:
        none
    '''
    #Prepping transitscore data and job data
    all_tract_data = tract_data(tracts_filepath, jobs_filepath)
    all_place_data = places_data(places_filepath)
    transit_score_added = get_transitscore(all_tract_data, all_place_data)

    #Merging all data with ACS and writing to pickle file
    acs5['tract_GEO_ID'] = acs5['GEO_ID'].apply(lambda x: x[9:])  
    full_df = pd.merge(acs5, transit_score_added, left_on='tract_GEO_ID', right_on='tract_GEO_ID')
    final_df = data_cleaning(full_df)
    final_df.to_pickle(filename)
