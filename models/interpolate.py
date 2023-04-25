"""Interpolate the TriOS-derived reflectance data to 1nm intervals
    so that the spectral resolution of the measured data corresponds to the spectral resolution of the modelled data.
    Use Cubic Spline interpolation"""

import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pylab as plt

trios_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/reflectance/trios_reflectance.csv')

ONE02_array = trios_reflectance['ONE02_MGT'].to_numpy()
ONE03_array = trios_reflectance['ONE03_MGT'].to_numpy()
ONE07_array = trios_reflectance['ONE07_MGT'].to_numpy()
ONE08_array = trios_reflectance['ONE08_MGT'].to_numpy()
ONE09_array = trios_reflectance['ONE09_MGT'].to_numpy()
ONE10_array = trios_reflectance['ONE10_MGT'].to_numpy()
ONE11_array = trios_reflectance['ONE11_MGT'].to_numpy()
ONE12_array = trios_reflectance['ONE12_MGT'].to_numpy()

RIM01_array = trios_reflectance['RIM01_MGT'].to_numpy()
RIM02_array = trios_reflectance['RIM02_MGT'].to_numpy()
RIM03_array = trios_reflectance['RIM03_MGT'].to_numpy()
RIM04_array = trios_reflectance['RIM04_MGT'].to_numpy()
RIM05_array = trios_reflectance['RIM05_MGT'].to_numpy()
RIM06_array = trios_reflectance['RIM06_MGT'].to_numpy()

LAG01B_array = trios_reflectance['LAG01B'].to_numpy()
ONE05A_array = trios_reflectance['ONE05A'].to_numpy()
ONE05B_array = trios_reflectance['ONE05B'].to_numpy()
ONE06A_array = trios_reflectance['ONE06A'].to_numpy()
ONE06B_array = trios_reflectance['ONE06B'].to_numpy()
RIM08A_array = trios_reflectance['RIM08A'].to_numpy()
RIM08B_array = trios_reflectance['RIM08B'].to_numpy()


trios_array = trios_reflectance['wavelength'].to_numpy()

x = trios_array
y0 = ONE02_array
y1 = ONE03_array
y2 = ONE07_array
y3 = ONE08_array
y4 = ONE09_array
y5 = ONE10_array
y6 = ONE11_array
y7 = ONE12_array
y8 = RIM01_array
y9 = RIM02_array
y10 = RIM03_array
y11 = RIM04_array
y12 = RIM05_array
y13 = RIM06_array
y14 = LAG01B_array
y15 = ONE05A_array
y16 = ONE05B_array
y17 = ONE06A_array
y18 = ONE06B_array
y19 = RIM08A_array
y20 = RIM08B_array

cs0 = CubicSpline(x, y0)
cs1 = CubicSpline(x, y1)
cs2 = CubicSpline(x, y2)
cs3 = CubicSpline(x, y3)
cs4 = CubicSpline(x, y4)
cs5 = CubicSpline(x, y5)
cs6 = CubicSpline(x, y6)
cs7 = CubicSpline(x, y7)
cs8 = CubicSpline(x, y8)
cs9 = CubicSpline(x, y9)
cs10 = CubicSpline(x, y10)
cs11 = CubicSpline(x, y11)
cs12 = CubicSpline(x, y12)
cs13 = CubicSpline(x, y13)
cs14 = CubicSpline(x, y14)
cs15 = CubicSpline(x, y15)
cs16 = CubicSpline(x, y16)
cs17 = CubicSpline(x, y17)
cs18 = CubicSpline(x, y18)
cs19 = CubicSpline(x, y19)
cs20 = CubicSpline(x, y20)

xs = np.arange(319, 951, 1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y0, label='Data', color='black', linestyle=':', linewidth=5)
ax.plot(xs, cs0(xs), label='Cubic spline interpolation', color='orange', linewidth=2)
ax.legend(loc='lower left', ncol=2)
ax.set_xlim(319, 800)
ax.set_ylim(0, 0.08)
plt.savefig('C:/Users/pirtapalola/Documents/iop_data/data/trios_reflectance_cubic_spline_interpolation',
            bbox_inches='tight', facecolor='w')
plt.show()

index_list = []
for z in range(0, 632):
    index_list.append(z)

ONE02_list = []
ONE02_benthic = []
for i in xs:
    n = cs0(i)
    ONE02_list.append(n)
for element in index_list:
    ONE02_benthic.append(float(ONE02_list[element]))

ONE03_list = []
ONE03_benthic = []
for i in xs:
    n = cs1(i)
    ONE03_list.append(n)
for element in index_list:
    ONE03_benthic.append(float(ONE03_list[element]))

ONE07_list = []
ONE07_benthic = []
for i in xs:
    n = cs2(i)
    ONE07_list.append(n)
for element in index_list:
    ONE07_benthic.append(float(ONE07_list[element]))

ONE08_list = []
ONE08_benthic = []
for i in xs:
    n = cs3(i)
    ONE08_list.append(n)
for element in index_list:
    ONE08_benthic.append(float(ONE08_list[element]))

ONE09_list = []
ONE09_benthic = []
for i in xs:
    n = cs4(i)
    ONE09_list.append(n)
for element in index_list:
    ONE09_benthic.append(float(ONE09_list[element]))

ONE10_list = []
ONE10_benthic = []
for i in xs:
    n = cs5(i)
    ONE10_list.append(n)
for element in index_list:
    ONE10_benthic.append(float(ONE10_list[element]))

ONE11_list = []
ONE11_benthic = []
for i in xs:
    n = cs6(i)
    ONE11_list.append(n)
for element in index_list:
    ONE11_benthic.append(float(ONE11_list[element]))

ONE12_list = []
ONE12_benthic = []
for i in xs:
    n = cs7(i)
    ONE12_list.append(n)
for element in index_list:
    ONE12_benthic.append(float(ONE12_list[element]))

RIM01_list = []
RIM01_benthic = []
for i in xs:
    n = cs8(i)
    RIM01_list.append(n)
for element in index_list:
    RIM01_benthic.append(float(RIM01_list[element]))

RIM02_list = []
RIM02_benthic = []
for i in xs:
    n = cs9(i)
    RIM02_list.append(n)
for element in index_list:
    RIM02_benthic.append(float(RIM02_list[element]))

RIM03_list = []
RIM03_benthic = []
for i in xs:
    n = cs10(i)
    RIM03_list.append(n)
for element in index_list:
    RIM03_benthic.append(float(RIM03_list[element]))

RIM04_list = []
RIM04_benthic = []
for i in xs:
    n = cs11(i)
    RIM04_list.append(n)
for element in index_list:
    RIM04_benthic.append(float(RIM04_list[element]))

RIM05_list = []
RIM05_benthic = []
for i in xs:
    n = cs12(i)
    RIM05_list.append(n)
for element in index_list:
    RIM05_benthic.append(float(RIM05_list[element]))

RIM06_list = []
RIM06_benthic = []
for i in xs:
    n = cs13(i)
    RIM06_list.append(n)
for element in index_list:
    RIM06_benthic.append(float(RIM06_list[element]))

LAG01B_list = []
LAG01B_benthic = []
for i in xs:
    n = cs14(i)
    LAG01B_list.append(n)
for element in index_list:
    LAG01B_benthic.append(float(LAG01B_list[element]))

ONE05A_list = []
ONE05A_benthic = []
for i in xs:
    n = cs15(i)
    ONE05A_list.append(n)
for element in index_list:
    ONE05A_benthic.append(float(ONE05A_list[element]))

ONE05B_list = []
ONE05B_benthic = []
for i in xs:
    n = cs16(i)
    ONE05B_list.append(n)
for element in index_list:
    ONE05B_benthic.append(float(ONE05B_list[element]))

ONE06A_list = []
ONE06A_benthic = []
for i in xs:
    n = cs17(i)
    ONE06A_list.append(n)
for element in index_list:
    ONE06A_benthic.append(float(ONE06A_list[element]))

ONE06B_list = []
ONE06B_benthic = []
for i in xs:
    n = cs18(i)
    ONE06B_list.append(n)
for element in index_list:
    ONE06B_benthic.append(float(ONE06B_list[element]))

RIM08A_list = []
RIM08A_benthic = []
for i in xs:
    n = cs19(i)
    RIM08A_list.append(n)
for element in index_list:
    RIM08A_benthic.append(float(RIM08A_list[element]))

RIM08B_list = []
RIM08B_benthic = []
for i in xs:
    n = cs20(i)
    RIM08B_list.append(n)
for element in index_list:
    RIM08B_benthic.append(float(RIM08B_list[element]))

ONE02_reflectance = pd.DataFrame()
ONE03_reflectance = pd.DataFrame()
ONE07_reflectance = pd.DataFrame()
ONE08_reflectance = pd.DataFrame()
ONE09_reflectance = pd.DataFrame()
ONE10_reflectance = pd.DataFrame()
ONE11_reflectance = pd.DataFrame()
ONE12_reflectance = pd.DataFrame()
RIM01_reflectance = pd.DataFrame()
RIM02_reflectance = pd.DataFrame()
RIM03_reflectance = pd.DataFrame()
RIM04_reflectance = pd.DataFrame()
RIM05_reflectance = pd.DataFrame()
RIM06_reflectance = pd.DataFrame()
LAG01B_reflectance = pd.DataFrame()
ONE05A_reflectance = pd.DataFrame()
ONE05B_reflectance = pd.DataFrame()
ONE06A_reflectance = pd.DataFrame()
ONE06B_reflectance = pd.DataFrame()
RIM08A_reflectance = pd.DataFrame()
RIM08B_reflectance = pd.DataFrame()

wavelength_319_951 = []

a = range(319, 951)
for q in a:
    wavelength_319_951.append(q)


def benthic_reflectance_function(reflectance_df, benthic_df):
    reflectance_df['wavelength'] = wavelength_319_951
    reflectance_df['benthic'] = benthic_df
    reflectance_df = reflectance_df[reflectance_df.wavelength > 404]
    reflectance_df = reflectance_df[reflectance_df.wavelength < 706]
    return reflectance_df


ONE02_refl = benthic_reflectance_function(ONE02_reflectance, ONE02_benthic)
ONE03_refl = benthic_reflectance_function(ONE03_reflectance, ONE03_benthic)
ONE07_refl = benthic_reflectance_function(ONE07_reflectance, ONE07_benthic)
ONE08_refl = benthic_reflectance_function(ONE08_reflectance, ONE08_benthic)
ONE09_refl = benthic_reflectance_function(ONE09_reflectance, ONE09_benthic)
ONE10_refl = benthic_reflectance_function(ONE10_reflectance, ONE10_benthic)
ONE11_refl = benthic_reflectance_function(ONE11_reflectance, ONE11_benthic)
ONE12_refl = benthic_reflectance_function(ONE12_reflectance, ONE12_benthic)
RIM01_refl = benthic_reflectance_function(RIM01_reflectance, RIM01_benthic)
RIM02_refl = benthic_reflectance_function(RIM02_reflectance, RIM02_benthic)
RIM03_refl = benthic_reflectance_function(RIM03_reflectance, RIM03_benthic)
RIM04_refl = benthic_reflectance_function(RIM04_reflectance, RIM04_benthic)
RIM05_refl = benthic_reflectance_function(RIM05_reflectance, RIM05_benthic)
RIM06_refl = benthic_reflectance_function(RIM06_reflectance, RIM06_benthic)
LAG01B_refl = benthic_reflectance_function(LAG01B_reflectance, LAG01B_benthic)
ONE05A_refl = benthic_reflectance_function(ONE05A_reflectance, ONE05A_benthic)
ONE05B_refl = benthic_reflectance_function(ONE05B_reflectance, ONE05B_benthic)
ONE06A_refl = benthic_reflectance_function(ONE06A_reflectance, ONE06A_benthic)
ONE06B_refl = benthic_reflectance_function(ONE06B_reflectance, ONE06B_benthic)
RIM08A_refl = benthic_reflectance_function(RIM08A_reflectance, RIM08A_benthic)
RIM08B_refl = benthic_reflectance_function(RIM08B_reflectance, RIM08B_benthic)

ONE02_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE02_trios.csv')
ONE03_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE03_trios.csv')
ONE07_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE07_trios.csv')
ONE08_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE08_trios.csv')
ONE09_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE09_trios.csv')
ONE10_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE10_trios.csv')
ONE11_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE11_trios.csv')
ONE12_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE12_trios.csv')
RIM01_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM01_trios.csv')
RIM02_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM02_trios.csv')
RIM03_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM03_trios.csv')
RIM04_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM04_trios.csv')
RIM05_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM05_trios.csv')
RIM06_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM06_trios.csv')
LAG01B_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/LAG01B_trios.csv')
ONE05A_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE05A_trios.csv')
ONE05B_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE05B_trios.csv')
ONE06A_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE06A_trios.csv')
ONE06B_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/ONE06B_trios.csv')
RIM08A_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM08A_trios.csv')
RIM08B_refl.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/trios/RIM08B_trios.csv')
