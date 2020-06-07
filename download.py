import censusdata
import pandas as pd
import geopandas as gpd
import pipeline

YEAR = 2018
STATE = '17'

DATA_COLS = {
         "B08301_001E": "commut_total",
         "B08301_010E": "commut_took_public_trans",
         "B08141_001E": "car_avail_total",
         "B08141_002E": "car_avail_none",
         "B17001_001E": "poverty_status_total",
         "B17001_002E": "poverty_status_yes",
         "C18108_001E": "disabl_status_total",
         "C18108_005E": "disabl_none_under_18",
         "C18108_009E": "disabl_none_18_to_64",
         "C18108_013E": "disabl_none_65_plus",
         "B02001_001E": "race_total",
         "B02001_002E": "race_white_alone",
         "B02001_003E": "race_black_alone",
         "B02001_005E": "race_asian_alone",
         "B03003_001E": "hispan_origin_total",
         "B03003_002E": "hispan_origin_no",
         "B03003_003E": "hispan_origin_yes",
         "B15003_001E": "educ_attnmt_total",
         "B15003_017E": "educ_attnmt_HS_reg",
         "B15003_018E": "educ_attnmt_HS_GED",
         "B15003_019E": "educ_attnmt_some_college_less_1_yr",
         "B15003_020E": "educ_attnmt_some_college_more_1_yr",
         "B15003_021E": "educ_attnmt_some_college_associates",
         "B15003_022E": "educ_attnmt_bachelors",
         "B15003_023E": "educ_attnmt_grad_masters",
         "B15003_024E": "educ_attnmt_grad_prof_degree",
         "B15003_025E": "educ_attnmt_grad_doctorate",
         "B19013_001E": "income_median",
         "B23001_001E": "emp_status_total",
         "B23001_006E": "emp_men_16_to_19_in_lbr_force",
         "B23001_013E": "emp_men_20_to_21_in_lbr_force",
         "B23001_020E": "emp_men_22_to_24_in_lbr_force",
         "B23001_027E": "emp_men_25_to_29_in_lbr_force",
         "B23001_062E": "emp_men_60_to_61_in_lbr_force",
         "B23001_069E": "emp_men_62_to_64_in_lbr_force",
         "B23001_074E": "emp_men_65_to_69_in_lbr_force",
         "B23001_079E": "emp_men_70_to_74_in_lbr_force",
         "B23001_084E": "emp_men_75_plus_in_lbr_force",
         "B23001_092E": "emp_women_16_to_19_in_lbr_force",
         "B23001_099E": "emp_women_20_to_21_in_lbr_force",
         "B23001_106E": "emp_women_22_to_24_in_lbr_force",
         "B23001_113E": "emp_women_25_to_29_in_lbr_force",
         "B23001_120E": "emp_women_30_to_34_in_lbr_force",
         "B23001_127E": "emp_women_35_to_44_in_lbr_force",
         "B23001_134E": "emp_women_45_to_54_in_lbr_force",
         "B23001_141E": "emp_women_55_to_59_in_lbr_force",
         "B23001_148E": "emp_women_60_to_61_in_lbr_force",
         "B23001_155E": "emp_women_62_to_64_in_lbr_force",
         "B23001_160E": "emp_women_65_to_69_in_lbr_force",
         "B23001_165E": "emp_women_70_to_74_in_lbr_force",
         "B23001_170E": "emp_women_75_plus_in_lbr_force",
         "B23025_003E": "total_in_labor_force",
         "B08012_001E": "commute_time_total",
         "B08012_002E": "commute_time_less_5_min",
         "B08012_003E": "commute_time_5_to_9_min",
         "B08012_004E": "commute_time_10_to_14_min",
         "B08012_005E": "commute_time_15_to_19_min",
         "B08012_006E": "commute_time_20_to_24_min",
         "B08012_007E": "commute_time_25_to_29_min",
         "B08012_008E": "commute_time_30_to_34_min",
         "B08012_009E": "commute_time_35_to_39_min",
         "B08012_010E": "commute_time_40_to_44_min",
         "B08012_011E": "commute_time_45_to_59_min",
         "B08012_012E": "commute_time_60_to_89_min",
         "B08012_013E": "commute_time_90_plus_min",
         "B12001_001E": "marital_status_total",
         "B12001_004E": "marital_status_men_married",
         "B12001_013E": "marital_status_women_married",
         "B09019_001E": "household_type_total",
         "B09019_027E": "household_type_men_liv_alone",
         "B09019_030E": "household_type_women_liv_alone",
         "C24070_001E": "industry_emp_total",
         "C24070_029E": "industry_self_emp_incorp",
         "C24070_071E": "industry_self_emp_non_incorp",
         "B07013_001E": "home_own_status",
         "B07013_002E": "home_own_yes",
         "B07013_003E": "home_rent_yes",
         "GEO_ID": "GEO_ID"
}

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
    acs5_processed = acs5.drop(list(CENSUS_DATA_COLS.values()), axis=1)

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

    return df
