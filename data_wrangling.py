import pandas as pd
import geopandas as gpd
from pandas.io.json import json_normalize
import requests 

tracts_filepath = 'data_sources/shape_tracts/tl_2018_17_tract.shp'
places_filepath = 'data_sources/shape_places/tl_2018_17_place.shp'
jobs_filepath = 'data_sources/il_jobs_by_tract_2017.csv'


def get_transitscore_data(lat, lng, city):
    '''
    Gets data for one location.
    
    Inputs:
        lat, lng (str)
    Returns:
        (json)
    '''
    rv = None
    
    #key = '4c4eb18a1eea25128110eaf683aefab4' #uchicago
    #key = 'ffd1c56f9abcf84872116b4cc2dfcf31' #gmail
    #key = 'b72221d8763203418d081f140357696e' #yahoo
    #key = 'a311336c201153d1963484f73744a740' #aol
    #key = '37b740e057b5b21f25e0d013026d66b4' #truman
    #key = 'b8e1215244ca3c87e2018dfa731744b5' #outlook
    key = 'b3cca8bcce0f65cd3fe82fb66f7a63c9' #protonmail
    
    url = 'https://transit.walkscore.com/transit/score/?lat={}&lon={}&city={}&state=IL&wsapikey={}'.format(lat, lng, city, key)
    r = requests.get(url)
    if not r.status_code==400:
        rv = r.json()
    print(r)
    print(r.text)
    
    return rv


def create_transitscore_datalist(df):
    '''
    Turns list of jsons into a dataframe
    
    Inputs:
        acsse data frame (pandas Dataframe)
    Returns:
        list containing transit score for all places for which it is available (list of json)
    '''
    datalist = []
    
    for row in df.itertuples():
        print(row.centroid_lat, row.centroid_lng, row.place_name)
        json = get_transitscore_data(row.centroid_lat, row.centroid_lng, row.place_name)
        if not json is None:
            json['tract_GEO_ID'] = row.tract_GEO_ID
            json['centroid_lat'] = row.centroid_lat
            json['centroid_lng'] = row.centroid_lng
            datalist.append(json)

    return datalist


def create_transitscore_dataframe(transitscore_datalist):
    '''
    Given the list of json objects containing transit score data, 
    create dataframe.

    Inputs:
        (list of json)
    Returns:
        (pandas DataFrame)
    '''    
    df = pd.DataFrame.from_dict(json_normalize(datalist), orient='columns')
    df = df[['tract_GEO_ID', 'transit_score']]
    grouped = df.groupby('tract_GEO_ID').mean()
    
    return grouped


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
    tracts = gpd.read_file(tracts_filepath)
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
    all_tract_data = pd.merge(tracts, jobs, how='inner', left_on='tract_GEO_ID', right_on='job_tract_GEO_ID')
    
    return all_tract_data


def place_data(places_filepath):
    '''
    Loads and cleans places shapefiles.

    Inputs:
        places_filepath (str)
    Outputs:
        (geopandas DataFrame)
    '''
    places = gpd.read_file(places_filepath)
    all_places_data = places[['GEOID', 'NAME', 'NAMELSAD', 'geometry']] \
                            .rename(columns={'GEOID': 'place_GEO_ID', 'NAME': 'place_name',
                           'NAMELSAD': 'place_name_and_type'})
    return all_places_data


def add_transitscore(all_tracts_data, all_places_data):
    '''
    Spatially joins tracts and places.
    Gets transit score for census tracts.
    Merges tract data and transit score data.

    Inputs:
        tracts, places (geopandas DataFrame)
    Outputs:
        (pandas DataFrame)
    '''
    #Merging tracts and places
    tracts_places = gpd.sjoin(tracts, places, how="inner", op="intersects")

    #Getting transit score data (may need to split tracts_place because of quota)
    datalist = create_transitscore_datalist(tracts_places)
    transitscore_by_tract = create_transitscore_dataframe(datalist)

    df = tracts.merge(transitscore_by_tract, on='tract_GEO_ID')
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
    df = df._get_numeric_data()
    df[df < 0] = np.nan
    df = df.drop(['year', 'centroid_lng', 'centroid_lat'], axis=1)

    return df


def go(acs5, tracts_filepath, places_filepath, jobs_filepath, pickle_filename=None):
    '''
    Executes all steps to take pulled ACS data to final dataframe for model 
    selection.

    Inputs:
        acs5 (pandas DataFrame)
        tracts_filepath, places_filepath, jobs_filepath, pickle_filename (str)
    Outputs:
        (pandas DataFrame)
    '''
    #Prepping transitscore data and job data
    all_tract_data = tract_data(tracts_filepath, jobs_filepath)
    all_place_data = place_data(places_filepath)
    transit_score_added = add_transitscore(all_tract_data, all_place_data)

    #Merging all data with ACS and writing to pickle file
    acs5['tract_GEO_ID'] = acs5['GEO_ID'].apply(lambda x: x[9:])  
    full_df = pd.merge(acs5, transit_score_added, on='tract_GEO_ID')
    final_df = data_cleaning(full_df)
    #final_df.to_pickle(filename)
    return final_df
