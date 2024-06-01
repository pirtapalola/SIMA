import numpy as np
import rasterio
from matplotlib import pyplot as plt
import matplotlib as mpl

import earthpy as et
import earthpy.plot as ep
import earthpy.spatial as es

ortho = rasterio.open('C:/Users/kell5379/Documents/Data/rimatuF1.tif')
ortho_arr = ortho.read()
fig, ax = plt.subplots(1, 2, figsize=(20, 20))

# plot
ep.plot_rgb(ortho_arr, ax=ax[0], rgb=[1, 2, 3], title="Red Green Blue", stretch=True)
ep.plot_rgb(ortho_arr, ax=ax[1], rgb=[3, 2, 1], title="NIR Green Blue", stretch=True)
plt.show()
