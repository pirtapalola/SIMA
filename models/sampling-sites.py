import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Read the csv file
site_locations = pd.read_csv('C:/seascape_data/sampling_sites_2022.csv',
                             engine='python',
                             usecols=['Unique_ID', 'Longitude', 'Latitude'],
                             index_col='Unique_ID')

# Create point geometries
site_geometry = gpd.points_from_xy(site_locations['Longitude'],
                                   site_locations['Latitude'], crs='EPSG:4326')

# Create a pandas geodataframe
site_gdf = gpd.GeoDataFrame(site_locations, geometry=site_geometry)
print(site_gdf)

# Test plotting a shapefile
onetahi_motu = gpd.GeoDataFrame.from_file('C:/Users/pirtapalola/Documents/'
                                          'Data/GIS/Courtney/Tetiaroa_Motu_Shapefiles/'
                                          'Tetiaroa_Motu_Shapefiles/Onetahi.shp')
onetahi_motu.plot()
plt.show()
