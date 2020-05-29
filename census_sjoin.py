import geopandas as gpd
# read shapefiles
blockgroups = gpd.read_file('tl_2019_17_bg.shp')
places = gpd.read_file('tl_2019_17_place.shp')
# feel free to inspect columns of original datasets to see if there is anything of interest
# merging place names and blockgroup geoid's, so only keeping bare minimum
places = places[['NAME', 'NAMELSAD', 'geometry']]
blockgroups = blockgroups[['GEOID', 'geometry']]
# merge dataset
blockgroups_places = gpd.sjoin(blockgroups, places, how="inner", op="intersects")