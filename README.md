# SIMA (Simulation-based Inference for Marine Analysis)

![Continuous Integration build in GitHub Actions](https://github.com/pirtapalola/seascapeRS/workflows/CI/badge.svg?branch=main)



This GitHub repository provides tools for the application of simulation-based inference to marine remote sensing.

**Please consider citing the associated research article and data repository:**
- Palola, P., Theenathayalan, V., Schröder, C., Martinez-Vicente, V., Collin, A., Wright, R., Ward, M., Thomson, E., Lopez-Garcia, P., Hochberg, E., Malhi, Y., and Wedding, L. (*In Review*). Simulation-based inference advances water
  quality mapping in shallow coral reef environments.
- *Open Science Framework* data repository: SBI_marine_remote_sensing (https://osf.io/pcdgv)



# Description of tools available in this repository

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



# Acknowledgements and funding statement


**This research leverages sbi, a PyTorch package for simulation-based inference, developed by Tejero-Cantero et al. (2020):**
- Tejero-Cantero, Á., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P. J., Greenberg, D. S., & Macke, J. H. (2020). sbi: A toolkit for simulation-based inference (v0.12.1). *Zenodo*. https://doi.org/10.5281/zenodo.3993098
- Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P., Greenberg, D., & Macke, J. (2020). sbi: A toolkit for simulation-based inference. Journal of Open Source Software, 5(52), 2505. https://doi.org/10.21105/joss.02505


**Acknowledgements:**

Warm thanks to Frank, Hinano, and Temakehu Murphy, Lusiano Kolokilagi, Vairupe Huioutu Pater, Tuterai Apuarii, Courtney Stuart, Kaya Malhi, Dr Benoît Stoll, Dr Claudia Giardino, Dr Monica Pinardi, Dr Stuart Painter, Lily Zhao, Louise-Océane Delion, Dr Marina Schneider, and Tetiaroa Society for their help and support during the field campaign. The authors would also like to thank Professor Nick Graham, Dr Casey Benkwitt, Dr Jayna DeVore, Dr Hannah Epstein, Dr Thomas Jackson, Dr Aser Mata, Dr Robbie Ramsay, Jennifer Appoo, and Gurjeet S. Singh for their advice during the early stages of this research project.


**Funding statement:**

This work was supported by the Bertarelli Foundation as part of the Bertarelli Programme in Marine Science and the Osk. Huttunen Foundation.
