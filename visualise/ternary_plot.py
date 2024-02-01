import matplotlib.pyplot as plt
import pandas as pd
import mpltern
from mpltern.datasets import get_triangular_grid

# Specify file location
path = "C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/Absorption/" \
       "absorption_ternary_plot.csv"

# Create a pandas dataframe
absorption_df = pd.read_csv(path)

# Create lists
sample_IDs = absorption_df["Unique_ID"]
sample_ID = sample_IDs[:5]
phy = absorption_df["a_phy_440"]
cdom = absorption_df["a_CDOM_440"]
nap = absorption_df["a_nap_440"]
total = absorption_df["a_total_440"]

# Calculate the fractions
phy_fraction = [x / y for x, y in zip(phy, total)]
cdom_fraction = [x / y for x, y in zip(cdom, total)]
nap_fraction = [x / y for x, y in zip(nap, total)]

# Plot the figure
t, l, r = get_triangular_grid()
fig = plt.figure(figsize=(10.8, 4.8))
ax = fig.add_subplot(121, projection='ternary')
ax.triplot(t, l, r)

for x in range(len(sample_ID)):
       for i in sample_ID:
              ax.scatter(phy_fraction[x], cdom_fraction[x], nap_fraction[x], alpha=0.5, label=i)

ax.legend()

# ax.scatter(phy_fraction[1], cdom_fraction[1], nap_fraction[1], color="blue", marker="x")
print(phy_fraction[1], cdom_fraction[1], nap_fraction[1])
plt.show()
