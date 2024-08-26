# seascapeRS

![Continuous Integration build in GitHub Actions](https://github.com/pirtapalola/seascapeRS/workflows/CI/badge.svg?branch=main)

**Tools available in models:**
- *apply_model.py:* Conduct inference on field data.
- *check_0.py:* Conduct inference on simulated data.
- *check_missing_files.py:* Check the simulation output and find missing files.
- *evaluate.py:*  Assess the performance of the inference scheme by calculating coverage probability.


**Tools available in models/benthic:**
- *read_mat_file.py*: Read a mat file and save the data into a csv file.
- *mean_global_values.py:* Calculate the average reflectance of a single benthic cover type from a large dataset of field measurements.
- *benthic_input.py:* Replace the default benthic reflectance data file in EcoLight with a custom file.

**Tools available in models/hyperspectral_preprocessing:**
- *radiometer_calibration.py:* Apply an in-water calibration factor to the hyperspectral data measured with the TriOS RAMSES radiometers.
- *read_dat_files.py:* Read the radiance/irradiance data exported from the MSDA_EX software and save into a csv file.
- *trios_interpolation.py:* Apply cubic spline interpolation to the hyperspectral reflectance data calculated from TriOS RAMSES radiometric measurements.
- *water_surface_correction.py:* Calculate just-above water reflectance from just-below water reflectance.

**Tools available in models/priors:**
- *dist_plotted.py:* Visualise the prior distributions.
- *inference.py:* Sample the prior distributions to create parameterisations for the simulator.
- *summary_statistics.py:* Calculate summary statistics from samples drawn from the prior distributions.

**Tools available in models/simulation_output:**
- *noise.py:* Add Gaussian noise to the spectral data.
- *output_get_values.py:* Create a csv file storing the EcoLight output in a correct format.
- *output_processing_2.py:* Extract reflectance from the csv files containing the EcoLight output.
- *select_bands.py:* Select specific spectral bands.

**Tools available in models/simulation_setup:**
- *check_setup.py:* Check the EcoLight set-up files by identifying differences between two files line-by-line.
- *filter_setup_files.py:* Filter through EcoLight set-up files given a condition.
- *runlistHL.py:* Create the runlist.txt file for the Ecolight simulations.
- *setup_bottom.py:* Change the bottom reflectance file name in the EcoLight setup files.
