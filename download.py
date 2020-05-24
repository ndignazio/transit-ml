import censusdata
import pandas as pd
# Example: downloads names and FIPS codes for all counties in the state of Illinois
geographies = censusdata.geographies(censusdata.censusgeo([('state', '17'), 
                                                           ('county', '*')]), 
                                    'acs5', 2015)
<<<<<<< HEAD
# downloads county-level demographic data for all counties in the state of 
# Illinois using 1-year estimates (more recent, less precise)
acs1_county = censusdata.download("acs1", 2018, 
                                  censusdata.censusgeo([("state", "17"), 
                                                        ("county", "*")]), 
                                  ["B02001_001E",
                                   "B02001_002E",
                                   "B02001_003E",
                                   "B02001_004E",
                                   "B02001_005E",
                                   "B02001_007E",
                                   "B02001_008E",
                                   "B03003_002E",
                                   "B19013_001E",
                                   "B08513_001E",
                                   "B08513_002E",
                                   "B08513_003E",
                                   "B08513_004E",
                                   "B08513_005E",
                                   "B08513_025E",
                                   "B08513_026E",
                                   "B08513_027E",
                                   "B08513_028E",
                                   "B08513_029E",
                                   "B08301_010E",
                                   "B08301_011E",
                                   "B08301_012E",
                                   "B08301_013E",
                                   "B08301_014E",

                                    "GEO_ID"])
# downloads tract-level data for all tracts in the state of Illinois 
# using 5-year estimates (less recent, more precise)
acs5_tract = censusdata.download("acs5", 2015, 
                                 censusdata.censusgeo([("state", "17"), ("tract", "*")]), 
                                  ["B02001_001E",
                                   "B02001_002E",
                                   "B02001_003E",
                                   "B02001_004E",
                                   "B02001_005E",
                                   "B02001_007E",
                                   "B02001_008E",
                                   "B03003_002E",
                                   "B19013_001E",
                                   "B08513_001E",
                                   "B08513_002E",
                                   "B08513_003E",
                                   "B08513_004E",
                                   "B08513_005E",
                                   "B08513_025E",
                                   "B08513_026E",
                                   "B08513_027E",
                                   "B08513_028E",
                                   "B08513_029E",
                                   "B08301_010E",
                                   "B08301_011E",
                                   "B08301_012E",
                                   "B08301_013E",
                                   "B08301_014E", "GEO_ID"])

columns = {
         "B02001_001E": "Total Population", "B02001_002E": "White Population", 
         "B02001_003E": "Black/African American Population", 
         "B02001_004E": "American Indian/Alaska Native Population",
         "B02001_005E": "Asian Population",
         "B02001_007E": "Other Races Population",
         "B02001_008E": "Two or More Races Population",
         "B03003_002E": "Not Hispanic or Latino Population",
         "B03003_003E": "Hispanic or Latino Population",
         "B19013_001E": "Median Income", "B08513_001E": "Means of Transport to Work: Total"
         "B08513_002E": "Speak English Only", 
         "B08513_003E": "Speak Spanish",
         "B08513_004E": "Speak English 'Very Well'",
         "B08513_005E": "Speak English 'Less Than Very Well'", "B08513_006E": "Speak Other Languages",
         # not sure if 025-030 are necessary?
         "B08513_025E": "Took Public Transport", "B08513_026E": "Public Transport: Speak English",
         "B08513_027E": "Public Transport: Speak Spanish", 
         "B08513_028E": "Public Transport: Speak Spanish/English Very Well"
         "B08513_029E": "Public Transport: Speak Spanish/English Less Than Very Well"
         "B08513_030E": "Public Transport: Speak Other Languages",
         "B08301_010E": "Took Public Transport",
         "B08301_011E": "Took Public Transport: Bus or Trolley Bus",
         "B08301_012E": "Took Public Transport: Streetcar/Trolley car"
         "B08301_013E": "Took Public Transport: Subway/Elevated",
         "B08301_014E": "Took Public Transport: Railroad"
}
acs1_county = acs1_county.rename(columns=columns)
acs5_tract = acs5_tract.rename(columns=columns)
=======
>>>>>>> eab83b6f1a3683eb055ff28c3158154ffc5ea42d

# using 1-year supplemental estimates
acsse = censusdata.download("acsse", 2018, censusdata.censusgeo([("state", "17"), 
                                                                ("place", "*")]), 
                                                               ["K200101_001E", # Total Population
                                                               "K200102_001E", # under 18 years old
                                                               "K200104_003E", # 18-24 years old
                                                               "K200104_004E", # 25-34 years old
                                                               "K200104_005E", # 35-44 years old
                                                               "K200104_006E", # 45-54 years old
                                                               "K200104_007E", # 55-64 years old
                                                               "K200104_008E", # over 65
                                                               "K201701_002E", # income in past 12 months below poverty level
                                                               "K201801_002E", # disability status: with a disability
                                                               "K201902_001E", # median household income in past 12 months
                                                                "GEO_ID"])
# check whether data includes cities with population greater than 65000
acsse[acsse['K200101_001E'] > 65000]
# yes, it does; should yield 18 rows
