# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

"""STEP 1. Read the parameter combinations from a csv file."""

csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                'Jan2024_lognormal_priors/Ecolight_parameter_combinations.csv'
combinations = pd.read_csv(csv_file_path)


# The data is a 100 x 5 matrix (100 simulation runs and 5 parameters)
# Choose columns
x = combinations["phy"]
y = combinations["cdom"]
z = combinations["spm"]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color representing the third dimension
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', edgecolors='k', s=10, label='Samples')

# Highlight specific samples (rows 55 and 94)
highlight_indices = [55, 94]
highlight_x = x.iloc[highlight_indices]
highlight_y = y.iloc[highlight_indices]
highlight_z = z.iloc[highlight_indices]
print(highlight_x, highlight_y, highlight_z)

# ax.scatter(highlight_x, highlight_y, highlight_z, c='red', marker='X', s=100, label='Highlighted Samples')

# Add colorbar for reference
cbar = fig.colorbar(scatter, ax=ax, label='NAP', pad=0.2)

# Adjust label positions
ax.set_xlabel('Phytoplankton', labelpad=10)
ax.set_ylabel('CDOM', labelpad=10)
ax.set_zlabel('NAP', labelpad=10)

# Show the plot
plt.show()
