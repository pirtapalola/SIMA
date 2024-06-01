
import rasterio
from pyproj import CRS, Transformer

file_path = 'C:/Users/kell5379/Documents/Data/rimatuF1.tif'


# Function to convert lat/lon to image coordinates
def latlon_to_image_coords(lat, lon, dataset):
    # Get the transform and CRS from the dataset
    transform = dataset.transform
    crs = dataset.crs

    # Define projection transformation using the new method
    wgs84 = CRS.from_epsg(4326)  # WGS84
    tif_proj = CRS.from_wkt(crs.to_wkt())  # TIF file projection

    # Create a transformer object
    transformer = Transformer.from_crs(wgs84, tif_proj, always_xy=True)

    # Transform coordinates
    x, y = transformer.transform(lon, lat)

    # Convert to image coordinates
    row, col = ~dataset.transform * (x, y)

    return int(row), int(col)


# Open the TIF file

with rasterio.open(file_path) as dataset:
    # Specify the latitude and longitude
    latitude = -17.02392203
    longitude = -149.56122308

    # Convert lat/lon to image coordinates
    row, col = latlon_to_image_coords(latitude, longitude, dataset)

    # Print debug information
    print(f"Latitude: {latitude}, Longitude: {longitude}")
    print(f"Transformed coordinates: x={col}, y={row}")
    print(f"Image dimensions: width={dataset.width}, height={dataset.height}")

    # Check if the coordinates are within the image bounds
    if 0 <= row < dataset.height and 0 <= col < dataset.width:
        # Read the values of the 10 bands at the specified location
        values = dataset.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], window=((row, row + 1), (col, col + 1)))

        # Since we are reading a single pixel, remove the extra dimension
        values = values[:, 0, 0]

        print("Values at the specified location:", values)
    else:
        print("The specified coordinates are out of the image bounds.")

