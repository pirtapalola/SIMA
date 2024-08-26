# seascapeRS

![Continuous Integration build in GitHub Actions](https://github.com/pirtapalola/seascapeRS/workflows/CI/badge.svg?branch=main)

**Tools available in models/benthic:**
- *read_mat_file.py*: Read a mat file and save the data into a csv file.
- *mean_global_values.py:* Calculate the average reflectance of a single benthic cover type from a large dataset of field measurements.
- *benthic_input.py:* Replace the default benthic reflectance data file in EcoLight with a custom file.

**Tools available in models/hyperspectral_preprocessing:**
- *radiometer_calibration.py:* Apply an in-water calibration factor to the hyperspectral data measured with the TriOS RAMSES radiometers.
- *read_dat_files.py:* Read the radiance/irradiance data exported from the MSDA_EX software and save into a csv file.
- *trios_interpolation.py:* Apply cubic spline interpolation to the hyperspectral reflectance data calculated from TriOS RAMSES radiometric measurements.
- *water_surface_correction.py:* Calculate just-above water reflectance from just-below water reflectance.
