import pandas as pd
import geopandas as gpd

# Read the csv file
site_locations = pd.read_csv('C:/Users/pirtapalola/seascapeRS/data/LOCAR_Site_Information.csv',
                             skipfooter=11, engine='python',
                             usecols=['Site Code', 'Longitude', 'Latitude'],
                             index_col='Site Code')

# Create point geometries
site_geometry = gpd.points_from_xy(site_locations['Longitude'],
                                   site_locations['Latitude'], crs='EPSG:4326')

# Create a pandas geodataframe
site_gdf = gpd.GeoDataFrame(site_locations, geometry=site_geometry)
print(site_gdf)
