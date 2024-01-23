"""

Conduct dimensionality reduction with PCA and plot the results in a scatter plot.

"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Dataset of 3000 spectra
simulated_spectra = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                                "Dec2023_lognormal_priors/simulated_reflectance.csv")

# Dataset of 17 spectra
measured_spectra = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                               "In_water_calibration_2022/above_water_reflectance_2022.csv")

# Combine the datasets
# all_data = pd.concat([simulated_spectra, measured_spectra], axis=1)
all_data = np.concatenate((simulated_spectra, measured_spectra), axis=1)

# Apply PCA with n_components=2 to the combined dataset
pca = PCA(n_components=2, random_state=42)
reduced_data = pca.fit_transform(all_data)

# Create a scatter plot
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='o')

# Add labels and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')

# Show the plot
plt.show()
