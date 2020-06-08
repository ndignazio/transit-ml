import censusdata
import pandas as pd
import geopandas as gpd
import pipeline

YEAR = 2018
STATE = '17'

# reading JSON file of census data columns
with open('CENSUS_DATA_COLS.json') as f:
    DATA_COLS = json.load(f)

def compile_and_merge_data():
    '''
    Wrapper function to
        (1) Compile DataFrame with Census data,
        (2) Engineer Census-related features and target,
        (3) Add additional features (transit score, job density, and population
            density), and
        (4) Clean/process data to create final dataframe

    Output:
        A pandas DataFrame that can be used to build a model
    '''
    acs5 = compile_acs_data(YEAR, STATE, DATA_COLS)
    processed_acs5 = create_census_features_and_target(acs5)

    return merge_data_sources(processed_acs5)


def compile_acs_data(year, state, data_cols):
    '''
    Downloads 5-year American Community Survey data from Census API,
        creates features for model, and removes original downloaded attributes

    Inputs:
        year (integer): year of data to be downloaded
        state (string): encoding of state for which to pull data
        data_cols (dict): keys are the columns (strings) to pull from the
            Census API, values are labels (strings) that should be given to
            the columns in the analysis

    Output:
        A pandas DataFrame with the necessary data for the analysis

    '''
    return pipeline.get_acs_5_data(year, state, data_cols)


def create_census_features_and_target(acs_df):
    '''
    Create DataFrame with Census-related features and target to build model

    Inputs:
        acs_df: A pandas DataFrame with data columns pulled from the Census API

    Output:
        A pandas DataFrame with
    '''

    acs5['commuting_ridership'] = acs5['commut_took_public_trans'] / acs5['commut_total']
    acs5['with_disability'] = (acs5['disabl_status_total'] - acs5['disabl_none_under_18'] -
                                   acs5['disabl_none_18_to_64'] - acs5['disabl_none_65_plus']) / \
                                   acs5['disabl_status_total']
    acs5['below_poverty_level'] = acs5['poverty_status_yes'] / acs5['poverty_status_total']
    acs5['carless'] = acs5['car_avail_none'] / acs5['car_avail_total']
    acs5['young_employed'] = (acs5['emp_men_16_to_19_in_lbr_force'] +
                             acs5['emp_men_20_to_21_in_lbr_force'] +
                             acs5['emp_men_22_to_24_in_lbr_force'] +
                             acs5['emp_men_25_to_29_in_lbr_force'] +
                             acs5['emp_women_16_to_19_in_lbr_force'] +
                             acs5['emp_women_20_to_21_in_lbr_force'] +
                             acs5['emp_women_22_to_24_in_lbr_force'] +
                             acs5['emp_women_25_to_29_in_lbr_force']) / \
                             acs5['emp_status_total']
    acs5['old_employed'] = (acs5["emp_men_60_to_61_in_lbr_force"] +
                            acs5["emp_men_62_to_64_in_lbr_force"] +
                            acs5["emp_men_65_to_69_in_lbr_force"] +
                            acs5["emp_men_70_to_74_in_lbr_force"] +
                            acs5["emp_men_75_plus_in_lbr_force"] +
                            acs5["emp_women_60_to_61_in_lbr_force"] +
                            acs5["emp_women_62_to_64_in_lbr_force"] +
                            acs5["emp_women_65_to_69_in_lbr_force"] +
                            acs5["emp_women_70_to_74_in_lbr_force"] +
                            acs5["emp_women_75_plus_in_lbr_force"]) / \
                            acs5['emp_status_total']

    acs5['women_in_labor_force'] =  (acs5['emp_women_16_to_19_in_lbr_force'] +
                                     acs5['emp_women_20_to_21_in_lbr_force'] +
                                     acs5['emp_women_22_to_24_in_lbr_force'] +
                                     acs5['emp_women_25_to_29_in_lbr_force'] +
                                     acs5["emp_women_30_to_34_in_lbr_force"] +
                                     acs5["emp_women_35_to_44_in_lbr_force"] +
                                     acs5["emp_women_45_to_54_in_lbr_force"] +
                                     acs5["emp_women_55_to_59_in_lbr_force"] +
                                     acs5["emp_women_60_to_61_in_lbr_force"] +
                                     acs5["emp_women_62_to_64_in_lbr_force"] +
                                     acs5["emp_women_65_to_69_in_lbr_force"] +
                                     acs5["emp_women_70_to_74_in_lbr_force"] +
                                     acs5["emp_women_75_plus_in_lbr_force"]) / \
                                     acs5['total_in_labor_force']
    acs5['high_school_diploma'] = acs5["educ_attnmt_HS_reg"] / acs5["educ_attnmt_total"]
    acs5['GED'] = acs5["educ_attnmt_HS_GED"] / acs5["educ_attnmt_total"]
    acs5['some_college'] = (acs5["educ_attnmt_some_college_less_1_yr"] +
                             acs5["educ_attnmt_some_college_more_1_yr"] +
                             acs5["educ_attnmt_some_college_associates"]) / \
                             acs5["educ_attnmt_total"]
    acs5['bachelors_degree'] = acs5["educ_attnmt_bachelors"] / acs5["educ_attnmt_total"]
    acs5['graduate_degree'] = (acs5["educ_attnmt_grad_masters"] +
                               acs5["educ_attnmt_grad_prof_degree"] +
                               acs5["educ_attnmt_grad_doctorate"]) / \
                               acs5["educ_attnmt_total"]
    acs5['less_than_high_school'] = (acs5["educ_attnmt_total"] -
                                     (acs5["educ_attnmt_HS_reg"] +
                                      acs5["educ_attnmt_HS_GED"] +
                                      acs5["educ_attnmt_some_college_less_1_yr"] +
                                      acs5["educ_attnmt_some_college_more_1_yr"] +
                                      acs5["educ_attnmt_some_college_associates"] +
                                      acs5["educ_attnmt_bachelors"] +
                                      acs5["educ_attnmt_grad_masters"] +
                                      acs5["educ_attnmt_grad_prof_degree"] +
                                      acs5["educ_attnmt_grad_doctorate"])) / \
                                      acs5["educ_attnmt_total"]
    acs5['pct_white'] = acs5["race_white_alone"] / acs5['race_total']
    acs5['pct_black'] = acs5['race_black_alone'] / acs5['race_total']
    acs5['pct_asian'] = acs5['race_asian_alone'] / acs5['race_total']
    acs5['pct_hispanic'] = acs5['hispan_origin_yes'] / acs5['hispan_origin_total']
    # is it possible to calculate nonhispanic white with just counts?
    acs5['commute_less_than_15'] = (acs5["commute_time_less_5_min"] +
                                    acs5["commute_time_5_to_9_min"] +
                                    acs5["commute_time_10_to_14_min"]) / \
                                    acs5['commute_time_total']
    acs5['commute_15to29'] = (acs5["commute_time_15_to_19_min"] +
                              acs5["commute_time_20_to_24_min"] +
                              acs5["commute_time_25_to_29_min"]) / \
                              acs5['commute_time_total']
    acs5['commute_30to44'] = (acs5["commute_time_30_to_34_min"] +
                              acs5["commute_time_35_to_39_min"] +
                              acs5["commute_time_40_to_44_min"]) / \
                              acs5['commute_time_total']
    acs5['commute_more_than_44'] = (acs5["commute_time_45_to_59_min"] +
                                    acs5["commute_time_60_to_89_min"] +
                                    acs5["commute_time_90_plus_min"]) / \
                                    acs5['commute_time_total']

    acs5['now_married'] = (acs5["marital_status_men_married"] +
                           acs5["marital_status_women_married"]) / \
                           acs5['marital_status_total']
    acs5['living_alone'] = (acs5["household_type_men_liv_alone"] +
                            acs5["household_type_women_liv_alone"]) / \
                            acs5['household_type_total']
    acs5['self_employed'] = (acs5["industry_self_emp_non_incorp"] +
                             acs5["industry_self_emp_incorp"]) / \
                            acs5["industry_emp_total"]
    acs5['median_income'] = acs5['income_median']
    acs5['renter_rate'] = acs5['home_rent_yes'] / acs5['home_own_status']
    # the following line drops original count values so that the ACS data
    # only contains the transformations above
    acs5_processed = acs5.drop(list(DATA_COLS.values()), axis=1)

    return acs5_processed


def merge_data_sources(acs5):
    '''
    Links acs5 data with transit score and job data. Calculates population
    density and job density.

    Inputs:
        acs5 (pandas DataFrame)

    Outputs:
        Full pandas DataFrame with transit score data
    '''
    #Extracting census tract ID
    acs5['tract_GEO_ID'] = acs5['GEO_ID'].apply(lambda x: x[9:])
    print('initial acs5 shape: {}'.format(acs5.shape[0]))
    print('inital number of colums: {}'.format(len(acs5.columns)))
    print("--------------------------------------------------------------")


    #Loading tracts
    tracts = gpd.read_file('shape_tracts/tl_2018_17_tract.shp')
    tracts = tracts[['GEOID', 'NAMELSAD', 'ALAND', 'geometry']] \
                        .rename(columns={'GEOID': 'tract_GEO_ID', 'NAMELSAD': 'tract_name',
                       'ALAND': 'tract_area'})

    #Loading places
    places = gpd.read_file('shape_places/tl_2018_17_place.shp')
    places = places[['GEOID', 'NAME', 'NAMELSAD', 'geometry']] \
                        .rename(columns={'GEOID': 'place_GEO_ID', 'NAME': 'place_name',
                       'NAMELSAD': 'place_name_and_type'})

    #Merging tracts and places
    tracts_places = gpd.sjoin(tracts, places, how="inner", op="intersects")

    #Merging acs data with traces/places
    df = pd.merge(acs5, tracts_places, left_on='tract_GEO_ID', right_on='tract_GEO_ID')
    print('df shape after merging with tracts_place: {}'.format(df.shape[0]))
    print('df number of columns: {}'.format(len(df.columns)))
    print("--------------------------------------------------------------")


    #Importing transit score csv and merging
    ts = pd.read_csv('transit_score.csv').rename(columns={'nearby_routes': 'num_nearby_routes', \
         'bus': 'num_bus_routes', 'rail': 'num_rail_routes', 'other': 'num_other_routes', \
         'city': 'city_from_ts', 'description': 'transit_description', 'summary': 'transit_summary', \
         'Lat': 'lat', 'Lon': 'lon'})
    ts['tsplace_GEO_ID'] = ts['GEO_ID'].apply(lambda x: x[9:])
    ts = ts.drop(columns=['censusgeo', 'Place_Type', 'state', 'GEO_ID'])
    df = pd.merge(df, ts, how='inner', left_on='place_GEO_ID', right_on='tsplace_GEO_ID')
    print('df shape after mergin with transit score: {}'.format(df.shape[0]))
    print('df number of columns: {}'.format(len(df.columns)))
    print("--------------------------------------------------------------")


    #Importing jobs by tract and merging
    jobs = pd.read_csv('il_jobs_by_tract_2017.csv')
    jobs = jobs[['id', 'label', 'c000']] \
            .rename(columns={'id': 'job_tract_GEO_ID', 'label': 'job_tract_label',
                             'c000': 'num_jobs'})
    jobs['job_tract_GEO_ID'] = jobs['job_tract_GEO_ID'].astype(str)
    df = pd.merge(jobs, df, how='inner', left_on='job_tract_GEO_ID', right_on='tract_GEO_ID')
    print('df shape after mergin with jobs data: {}'.format(df.shape[0]))
    print('df number of columns: {}'.format(len(df.columns)))
    print("--------------------------------------------------------------")
    jobs_cols = set(df.columns)

    #Averaging transit score for census tracts
    df = df.groupby('GEO_ID').mean().reset_index()
    print('df shape after grouping: {}'.format(df.shape[0]))
    print('df number of colums: {}'.format(len(df.columns)))
    print('df cols omitted becauseo of grouping: {}'.format(jobs_cols - set(df.columns)))
    print("--------------------------------------------------------------")

    #Calculating population density and job density
    df['job_density'] = df['num_jobs'] / ((df['tract_area'])/1000000)
    df['pop_density'] = df['race_total'] / ((df['tract_area'])/1000000)

    #Dropping rows with zero population
    index_names = df[df['race_total']==0].index
    df.drop(index_names , inplace=True)
    print('df num columns with zero population: {}'.format(len(df[df['race_total']==0])))

    #Changing NaN values to zero (occur for small census tracts)
    df = df.fillna(0)
    
    ### TWO CHANGES NATHAN ADDED ###
    # changing negative values(in median income column) to NaN
    df = df._get_numeric_data()
    df[df < 0] = np.nan
    
    #Dropping remaining irrelevant columns
    df = df.drop(['year', 'lat', 'lon', 'index_right', 'num_nearby_routes', 'num_bus_routes',
                        'num_rail_routes', 'num_other_routes'], axis=1)

    return df
