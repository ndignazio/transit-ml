import censusdata
import pandas as pd
# Example: downloads names and FIPS codes for all counties in the state of Illinois
geographies = censusdata.geographies(censusdata.censusgeo([('state', '17'), 
                                                           ('county', '*')]), 
                                    'acs5', 2015)

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
