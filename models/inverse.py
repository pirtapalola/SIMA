import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from lmfit import Parameters, Minimizer
from matplotlib import rcParams


closure_data = pd.read_csv("C:/Users/pirtapalola/Documents/iop_data/data/closure_experiment.csv")
sites = ['ONE07', 'ONE08', 'ONE09', 'ONE10', 'ONE11', 'ONE12',
         'RIM01', 'RIM02', 'RIM03', 'RIM04', 'RIM05', 'RIM06']
rrs = closure_data['ONE07_surface']
plt.plot(rrs)
# plt.show()


def forward1(params, x, data=None):
    phy = params['phy']
    cdom = params['cdom']
    nap = params['nap']
    zb = params['zb']
    theta_air = params['theta_air']

    # Phytoplankton absorption coefficient

    # Sub-surface solar zenith angle in radians
    # Refractive index of seawater (temperature = 20C, salinity = 25g/kg, light wavelength = 589.3 nm)
    water_refractive_index = 1.33784
    inv_refractive_index = 1.0 / water_refractive_index
    # Sub-surface viewing angle in radians
    theta_w = math.asin(inv_refractive_index * math.sin(math.radians(theta_air)))
    inv_cos_theta_w = 1.0 / math.cos(theta_w)
    # Diffuse attenuation coefficient
    kd_coefficient = inv_cos_theta_w




    kappa = [aw[x] + bbw[x] for x in range(len(aw))]
    u_var = [bbw[x] / kappa[x] for x in range(len(bbw))]
    rw = [(0.084 + 0.17 * n) * n for n in u_var]
    model = [rw[x] + (rb[x] - rw[x])
             * np.exp(-2 * kd[x] * zb) for x in range(len(rb))]
    if data is None:
        return model
    else:
        return [model[x]-data[x] for x in range(len(data))]
