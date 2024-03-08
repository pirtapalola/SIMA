from sbi import utils as utils
from sbi.analysis import pairplot, conditional_pairplot, conditional_corrcoeff
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML, Image
import pickle
import pandas as pd
import torch

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/loaded_posterior9.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Define an observation
# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/' 
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/checks/check_0/check0_x.csv')
x_array = simulated_reflectance.iloc[32]  # X contains the simulated spectra
x_obs = torch.tensor(x_array, dtype=torch.float32)  # Convert to tensors

# Sample the posterior and extract the first three parameters for visualization
posterior_samples = loaded_posterior.sample((1000,), x=x_obs)
posterior_samples_3d = posterior_samples[:, :3]

rc("animation", html="html5")

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim((0, 2))
ax.set_ylim((0, 2))
ax.set_zlim((0, 2))


def init():
    scatter = ax.scatter([], [], [], s=15, c="#2171b5", depthshade=False)
    return (scatter,)


def animate(frame):
    num_samples_vis = 1000
    scatter = ax.scatter(
        posterior_samples_3d[:num_samples_vis, 0],
        posterior_samples_3d[:num_samples_vis, 1],
        posterior_samples_3d[:num_samples_vis, 2],
        s=15,
        c="#2171b5",
        depthshade=False,
    )
    ax.view_init(20, frame)
    return (scatter,)

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=range(0, 360, 5), interval=150, blit=True
)

plt.close()

HTML(anim.to_html5_video())
