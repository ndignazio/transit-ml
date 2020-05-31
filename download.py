import censusdata
import pandas as pd
import pipeline

STATE = '17'
YEAR = 2018

CENSUS_DATA_COLS = {
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

acs5 = pipeline.get_acs_5_data(YEAR, STATE, CENSUS_DATA_COLS)

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


#LINKING ACS DATA WITH TRANSIT SCORE
def acs_transitscore(asc5):
    '''
    Links acs and transit score data by performign the following tasks.
        - Extracts blockgroup FIPS code
        - Loads shape files for blockgroup and places
        - Performs a spatial join on blockgroup and places
        - Merge acs data with the blockgroup_places geopandas DataFrame
        - Loads csv of transit scores for all available Illinois cities 
        - Merge with transit score csv
    (We should probably still rename some columns here.)
    Inputs:
        acs5 (pandas DataFrame)
    Outputs:
        (full pandas DataFrame with transit score data)
    '''
    acs5 = acs5.rename(columns={'index': 'censusgeo'})
    acs5['tr_GEOID'] = acs5['GEO_ID'].apply(lambda x: x[9:])

    tracts = gpd.read_file('shape_tracts/tl_2018_17_tract.shp')
    places = gpd.read_file('places_shape/tl_2018_17_place.shp')
    tracts = tracts[['GEOID', 'NAMELSAD', 'geometry']]
    places = places[['GEOID', 'NAME', 'NAMELSAD', 'geometry']]

    tracts_places = gpd.sjoin(tracts, places, how="inner", op="intersects")
    df = pd.merge(acs5, tracts_places, left_on='tr_GEOID', right_on='GEOID_left')
    ts = pd.read_csv('transit_score.csv')

    return pd.merge(df, ts, how='inner', left_on='NAME', right_on='city')


# acs5 = pipeline.get_acs_data(SURVEY, YEARS, state=STATE,
#                                    data_columns=list(COL_MAPPING.keys()),
#                                   data_aliases=COL_MAPPING)
# '''
#
# #CALLING API FOR BLOCKGROUP DATA
# geographies = censusdata.geographies(censusdata.censusgeo([('state', '17'),
#     ('county', '*')]), 'acs5', 2018)
#
# acs5 = pd.DataFrame()
# for v in geographies.values():
#     ( (_, _) , (_, county_code) ) = v.params()
#     df = censusdata.download("acs5", 2018, censusdata.censusgeo(
#         [("state", "17"), ("county", county_code), ("block group", "*")]),
#         CENSUS_DATA_COLS.keys()).reset_index()
#     acs5 = acs5.append(df, ignore_index=True)




# Example: downloads names and FIPS codes for all counties in the state of Illinois
# geographies = censusdata.geographies(censusdata.censusgeo([('state', '17'),
#                                                            ('county', '*')]),
#                                      'acs5', 2015)

# # downloads county-level demographic data for all counties in the state of
# # Illinois using 1-year estimates (more recent, less precise)
# acs1_county = censusdata.download("acs1", 2018,
#                                   censusdata.censusgeo([("state", "17"),
#                                                         ("county", "*")]),
#                                   ["B02001_001E",
#                                    "B02001_002E",
#                                    "B02001_003E",
#                                    "B02001_004E",
#                                    "B02001_005E",
#                                    "B02001_007E",
#                                    "B02001_008E",
#                                    "B03003_002E",
#                                    "B19013_001E",
#                                    "B08513_001E",
#                                    "B08513_002E",
#                                    "B08513_003E",
#                                    "B08513_004E",
#                                    "B08513_005E",
#                                    "B08513_025E",
#                                    "B08513_026E",
#                                    "B08513_027E",
#                                    "B08513_028E",
#                                    "B08513_029E",
#                                    "B08301_010E",
#                                    "B08301_011E",
#                                    "B08301_012E",
#                                    "B08301_013E",
#                                    "B08301_014E",
#
#                                     "GEO_ID"])
# # downloads tract-level data for all tracts in the state of Illinois
# # using 5-year estimates (less recent, more precise)
# acs5_tract = censusdata.download("acs5", 2015,
#                                  censusdata.censusgeo([("state", "17"), ("tract", "*")]),
#                                   ["B02001_001E",
#                                    "B02001_002E",
#                                    "B02001_003E",
#                                    "B02001_004E",
#                                    "B02001_005E",
#                                    "B02001_007E",
#                                    "B02001_008E",
#                                    "B03003_002E",
#                                    "B19013_001E",
#                                    "B08513_001E",
#                                    "B08513_002E",
#                                    "B08513_003E",
#                                    "B08513_004E",
#                                    "B08513_005E",
#                                    "B08513_025E",
#                                    "B08513_026E",
#                                    "B08513_027E",
#                                    "B08513_028E",
#                                    "B08513_029E",
#                                    "B08301_010E",
#                                    "B08301_011E",
#                                    "B08301_012E",
#                                    "B08301_013E",
#                                    "B08301_014E", "GEO_ID"])
#
# columns = {
#          "B02001_001E": "Total Population", "B02001_002E": "White Population",
#          "B02001_003E": "Black/African American Population",
#          "B02001_004E": "American Indian/Alaska Native Population",
#          "B02001_005E": "Asian Population",
#          "B02001_007E": "Other Races Population",
#          "B02001_008E": "Two or More Races Population",
#          "B03003_002E": "Not Hispanic or Latino Population",
#          "B03003_003E": "Hispanic or Latino Population",
#          "B19013_001E": "Median Income", "B08513_001E": "Means of Transport to Work: Total",
#          "B08513_002E": "Speak English Only",
#          "B08513_003E": "Speak Spanish",
#          "B08513_004E": "Speak English 'Very Well'",
#          "B08513_005E": "Speak English 'Less Than Very Well'", "B08513_006E": "Speak Other Languages",
#          # not sure if 025-030 are necessary?
#          "B08513_025E": "Took Public Transport", "B08513_026E": "Public Transport: Speak English",
#          "B08513_027E": "Public Transport: Speak Spanish",
#          "B08513_028E": "Public Transport: Speak Spanish/English Very Well"
#          "B08513_029E": "Public Transport: Speak Spanish/English Less Than Very Well"
#          "B08513_030E": "Public Transport: Speak Other Languages",
#          "B08301_010E": "Took Public Transport",
#          "B08301_011E": "Took Public Transport: Bus or Trolley Bus",
#          "B08301_012E": "Took Public Transport: Streetcar/Trolley car"
#          "B08301_013E": "Took Public Transport: Subway/Elevated",
#          "B08301_014E": "Took Public Transport: Railroad"
# }
# acs1_county = acs1_county.rename(columns=columns)
# acs5_tract = acs5_tract.rename(columns=columns)


# using 1-year supplemental estimates
# acsse = censusdata.download("acsse", 2018, censusdata.censusgeo([("state", "17"),
#                                                                 ("place", "*")]),
#                                                                ["K200101_001E", # Total Population
#                                                                "K200102_001E", # under 18 years old
#                                                                "K200104_003E", # 18-24 years old
#                                                                "K200104_004E", # 25-34 years old
#                                                                "K200104_005E", # 35-44 years old
#                                                                "K200104_006E", # 45-54 years old
#                                                                "K200104_007E", # 55-64 years old
#                                                                "K200104_008E", # over 65
#                                                                "K201701_002E", # income in past 12 months below poverty level
#                                                                "K201801_002E", # disability status: with a disability
#                                                                "K201902_001E", # median household income in past 12 months
#                                                                 "GEO_ID"])
# check whether data includes cities with population greater than 65000
# acsse[acsse['K200101_001E'] > 65000]
# yes, it does; should yield 18 rows
#
# ACSSE_DATA = {"K200101_001E": "Total Population",
#               "K200201_002E": "white_alone",
#               "K200201_003E":  "black_alone",
#               "K200201_004E": "native_alone",
#               "K200201_005E":  "asian_alone",
#               "K200201_006E": "hawaiian/pacific_islander",
#               "K200201_007E": "other_races",
#               "K200201_008E": "two_or_more_races",
#               "K200301_002E": "not_hispanic/latino",
#               "K200301_003E": "hispanic/latino",
#               "K200701_002E": "same_house_1_year_ago",
#               "K200701_003E": "moved_from_within_county",
#               "K200701_004E": "moved_from_diff_county_same_state",
#               "K200701_005E": "moved from diff state",
#               "K200701_006E": "moved_from_abroad",
#               "K200801_004E": "took_public_transit",
#               "K200801_001E": "total_means_of_transit",
#               "K200801_006E": "worked_from_home",
#               "K200901_001E": "total_household_type",
#               "K200901_002E": "family households",
#               "K200901_003E": "married_couple_family",
#               "K200901_007E": "nonfamily_households",
#               "K200901_008E": "nonfamily_householder_living_alone",
#               "K200901_009E": "nonfamily_householder_not_living_alone",
#               "K201102_002E": "households_with_person_60+",
#               "K201601_002E": "english_only_households",
#               "K201601_003E": "spanish_speaking_households",
#               "K201601_006E": "other_languages",
#               "K200102_001E": "under_18",
#               "K200104_003E": "18-24",
#               "K200104_004E": "25-34",
#               "K200104_005E": "35-44",
#               "K200104_006E": "45-54",
#               "K200104_007E": "55-64",
#               "K200104_008E": "over 65",
#               "K201701_002E": "income_in_past_12_months_below_poverty_level",
#               "K201801_002E": "with_disability",
#               "K201902_001E": "median_household_income_past_12_months",
#               "K202801_002E": "households_with_computer",
#               "K202801_004E": "computer_and_broadband",
#               "GEO_ID": "GEO_ID"}
