# seascapeRS

![Continuous Integration build in GitHub Actions](https://github.com/pirtapalola/seascapeRS/workflows/CI/badge.svg?branch=main)

**Tools available in models:**
- *apply_model.py:* Conduct inference on field data.
- *check_0.py:* Conduct inference on simulated data.
- *check_missing_files.py:* Check the simulation output and find missing files.
- *evaluate.py:*  Assess the performance of the inference scheme by calculating coverage probability.
- *modelSBI.py:* Train the neural density estimator and build the posterior.
- *posterior_predictive_check.py:* Posterior predictive check (part 1): sample from the posterior.
- *posterior_predictive_check2.py:* Posterior predictive check (part 2): plot the posterior predictive.
- *prior_predictive_check.py:* Plot the prior predictive.
- *sbc.py:* Conduct simulation-based calibration.
- *tools.py:* Tools to create a custom prior distribution.

**Tools available in models/hyperspectral_preprocessing:**
- *interpolation.py:* Apply cubic spline interpolation to the hyperspectral reflectance data calculated from TriOS RAMSES radiometric measurements.
- *water_surface_correction.py:* Calculate just-above water reflectance from just-below water reflectance.

**Tools available in models/priors:**
- *dist_plotted.py:* Visualise the prior distributions.
- *create_setup.py:* Sample the prior distributions to create parameterisations for the simulator.
- *wind.py:* Estimate the prior distribution for wind speed.

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

**Tools available in models/visualise:**
- *range_plot.py:* Produce a range plot to visualise the results of the inference.
- *scatter.py:* Create a 3D plot to visualise the parameter space.
- *scatter_plot.py:* Produce a scatter plot to visualise the results of the inference.
