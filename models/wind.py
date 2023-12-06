import copernicus_marine_client as copernicus_marine

# DS = copernicus_marine.open_dataset(dataset_id="cmems_obs-wind_glo_phy_nrt_l3-hy2b-hscat-asc-0.25deg_P1D-i")
# print(DS.data_vars)


# Set parameters
data_request = {
   "dataset_id": "cmems_obs-wind_glo_phy_nrt_l3-hy2b-hscat-asc-0.25deg_P1D-i",
   "longitude": [-6.17, -5.09],
   "latitude": [35.75, 36.29],
   "time": ["2022-01-01", "2022-01-31"],
   "variables": ["wind_speed"]
}

# Load xarray dataset
sst_l3s = copernicus_marine.open_dataset(
    dataset_id = data_request["dataset_id"],
    minimum_longitude = data_request["longitude"][0],
    maximum_longitude = data_request["longitude"][1],
    minimum_latitude = data_request["latitude"][0],
    maximum_latitude = data_request["latitude"][1],
    start_datetime = data_request["time"][0],
    end_datetime = data_request["time"][1],
    variables = data_request["variables"]
)

# Print loaded dataset information
print(sst_l3s)
