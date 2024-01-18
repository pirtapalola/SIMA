"""

Plot a horizontal barplot.

Last updated on 18 January 2024 by Pirta Palola

"""

# Import libraries
import matplotlib.pyplot as plt

# Create bars
bars = ["Surface runoff", "Submarine groundwater discharge", "Weathering", "Tides", "Currents", "Waves", "Upwelling",
        "Trophic interactions", "Atmospheric deposition", "Pelagic N$_2$ fixation", "Sediment microbial processes",
        "Sediment resuspension"]
no_studies = [48, 23, 1, 8, 9, 4, 5, 8, 4, 2, 9, 2]
height = [int(i/76*100) for i in no_studies]

# Assign colors
colors = ["#538d22", "#538d22", "#538d22", "#57cc99", "#3a7ca5", "#3a7ca5", "#3a7ca5", "#c7f9cc", "#deeff5",
          "#deeff5", "#ffea61", "#ffea61"]

# Reverse the order of the lists
bars = bars[::-1]
height = height[::-1]
colors = colors[::-1]

# Create horizontal bar plot
fig, ax = plt.subplots(figsize=(16, 6))
bars_plot = ax.barh(bars, height, color=colors)

# Add percentage labels
for bar, h in zip(bars_plot, height):
    plt.text(bar.get_width()+0.25, bar.get_y() + bar.get_height() / 2, f'{h}%', ha='left', va='center', fontsize=8)

# Add labels and title
plt.xlabel('Percentage of studies (%)')
plt.ylabel('Nutrient pathways')

# Adjust layout
plt.subplots_adjust(left=0.5)

# Show the plot
plt.show()
