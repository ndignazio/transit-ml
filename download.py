import censusdata
import pandas as pd
import geopandas as gpd
import pipeline
import json
import data_wrangling

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

    return data_wrangling.go(processed_acs5)


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


def create_census_features_and_target(acs5):
    '''
    Create DataFrame with Census-related features and target to build model

    Inputs:
        acs5: A pandas DataFrame with data columns pulled from the Census API

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
    

    return acs5_processed
