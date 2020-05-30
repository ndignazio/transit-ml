import censusdata
import pandas as pd
import geopandas as gpd
import pipeline

SURVEY = 'acs5'
STATE = '17'
YEARS = [2018]

ACSSE_DATA = {"K200101_001E": "Total Population",
              "K200201_002E": "white_alone",
              "K200201_003E":  "black_alone",
              "K200201_004E": "native_alone",
              "K200201_005E":  "asian_alone",
              "K200201_006E": "hawaiian/pacific_islander",
              "K200201_007E": "other_races",
              "K200201_008E": "two_or_more_races",
              "K200301_002E": "not_hispanic/latino",
              "K200301_003E": "hispanic/latino",
              "K200701_002E": "same_house_1_year_ago",
              "K200701_003E": "moved_from_within_county",
              "K200701_004E": "moved_from_diff_county_same_state",
              "K200701_005E": "moved from diff state",
              "K200701_006E": "moved_from_abroad",
              "K200801_004E": "took_public_transit",
              "K200801_001E": "total_means_of_transit",
              "K200801_006E": "worked_from_home",
              "K200901_001E": "total_household_type",
              "K200901_002E": "family households",
              "K200901_003E": "married_couple_family",
              "K200901_007E": "nonfamily_households",
              "K200901_008E": "nonfamily_householder_living_alone",
              "K200901_009E": "nonfamily_householder_not_living_alone",
              "K201102_002E": "households_with_person_60+",
              "K201601_002E": "english_only_households",
              "K201601_003E": "spanish_speaking_households",
              "K201601_006E": "other_languages",
              "K200102_001E": "under_18",
              "K200104_003E": "18-24",
              "K200104_004E": "25-34",
              "K200104_005E": "35-44",                                                               
              "K200104_006E": "45-54",
              "K200104_007E": "55-64",
              "K200104_008E": "over 65",
              "K201701_002E": "income_in_past_12_months_below_poverty_level",
              "K201801_002E": "with_disability",
              "K201902_001E": "median_household_income_past_12_months",
              "K202801_002E": "households_with_computer",
              "K202801_004E": "computer_and_broadband",
              "GEO_ID": "GEO_ID"}

COL_MAPPING = {
         "B02001_001E": "pop_total",
         "B02001_002E": "pop_white",
         "B02001_003E": "pop_black",
         "B02001_004E": "pop_native",
         "B02001_005E": "pop_asian",
         "B02001_007E": "pop_other_races",
         "B02001_008E": "pop_mult_races",
         "B03003_002E": "pop_non_hispanic",
         "B03003_003E": "pop_hispanic",
         "B19013_001E": "median_income",
         "B08513_001E": "means_transport_to_work_total",
         "B08513_002E": "speak_english_only",
         "B08513_003E": "speak_spanish",
         "B08513_004E": "speak_english_very_well",
         "B08513_005E": "speak_english_less_than_very_well",
         "B08513_006E": "speaks_other_languages",
         "B08301_010E": "took_public_transport",
         "B08301_011E": "took_public_transport_bus_or_trolley_bus",
         "B08301_012E": "took_public_transport_streetcar_or_trolleycar",
         "B08301_013E": "took_public_transport_subway_or_elevated",
         "B08301_014E": "took_public_transport_railroad",
         "GEO_ID": "GEO_ID"
}

'''
acs5 = pipeline.get_acs_data(SURVEY, YEARS, state=STATE,
                                   data_columns=list(COL_MAPPING.keys()),
                                  data_aliases=COL_MAPPING)
'''

#CALLING API FOR BLOCKGROUP DATA
geographies = censusdata.geographies(censusdata.censusgeo([('state', '17'),
    ('county', '*')]), 'acs5', 2018)

acs5 = pd.DataFrame()
for v in geographies.values():
    ( (_, _) , (_, county_code) ) = v.params()
    df = censusdata.download("acs5", 2018, censusdata.censusgeo(
        [("state", "17"), ("county", county_code), ("block group", "*")]), ["GEO_ID"]).reset_index()
    acs5 = acs5.append(df, ignore_index=True)


#LINKING ACS DATA WITH TRANSIT SCORE
def acs_transitscore(asc5_df):
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
    acs5_df (pandas DataFrame)
  Outputs:
    (full pandas DataFrame with )
  '''
acs5 = acs5.rename(columns={'index': 'censusgeo'})
acs5['bg_GEOID'] = acs5['GEO_ID'].apply(lambda x: x[9:])

blockgroups = gpd.read_file('bg_shape/tl_2018_17_bg.shp')
places = gpd.read_file('places_shape/tl_2018_17_place.shp')
blockgroups = blockgroups[['GEOID', 'NAMELSAD', 'geometry']]
places = places[['GEOID', 'NAME', 'NAMELSAD', 'geometry']]

blockgroups_places = gpd.sjoin(blockgroups, places, how="inner", op="intersects")
df = pd.merge(acs5, blockgroups_places, left_on='bg_GEOID', right_on='GEOID_left')
ts = pd.read_csv('transit_score.csv')

return pd.merge(df, ts, how='inner', left_on='NAME', right_on='city')

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
