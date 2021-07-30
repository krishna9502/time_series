import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from statsmodels.tsa.seasonal import seasonal_decompose

from mgc_data import MGC_generator

## real_data
file = h5py.File('./../../dataset_folder/real_data.hdf5', 'r')
# print(file.keys())
main_group = 'monash_forecast'
# print(file['monash_forecast'].keys())
# data_group = 'cif_2016_dataset'
data_group = 'nn5_daily_dataset_without_missing_values'
# print(file['monash_forecast']['cif_2016_dataset'].keys())
length = len(file[main_group][data_group].keys())

random_data_id = np.random.randint(low=0, high=length)
random_data = file[main_group][data_group][data_group+str(random_data_id)][:]

real_data = random_data

## synthetic data
mgc_generator = MGC_generator()
data = mgc_generator.generate_mgc_data(35)[1]
synthetic_data = data


# data = real_data
data = synthetic_data
dates = pd.date_range(start='1/1/2016',periods=len(data))
data = pd.Series(data, index = dates)
## seasonal-trend-residual decomposition
# Additive Decomposition
result_add = seasonal_decompose(data, model='additive', extrapolate_trend=1, period=30)
residual = data - result_add.trend

t = np.arange(100,1000)

out_array = np.zeros(900)
for i in range(10):
    T = i+1
    out_array += T**2 * np.sin(2*np.pi*t/T) 



## plots 
fig = plt.figure(constrained_layout= True)
gs = fig.add_gridspec(2, 2)
fig_ax1 = fig.add_subplot(gs[0,:])
fig_ax1.set_title('raw data')
fig_ax1.plot(out_array, color = 'blue')
# fig_ax1.plot(residual, color = 'red')
# fig_ax1.plot(result_add.trend, color = 'green')
# fig_ax1.plot(result_add.seasonal, color = 'black')
fig_ax2 = fig.add_subplot(gs[1,0])
fig_ax2.set_title('autocorrelation')
fig_ax2.acorr(out_array, maxlags = 20) 
fig_ax3 = fig.add_subplot(gs[1,1])
fig_ax3.set_title('periodogram')
fig_ax3.psd(out_array, Fs=2) 
plt.show()