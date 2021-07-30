import numpy as np
import h5py
import random
# import pandas as pd
# import matplotlib.pyplot as plt
import os
from tqdm import tqdm 
import datetime

from data_loader_forecast import convert_tsf_to_dataframe
from data_loader_regression import load_from_tsfile_to_dataframe

forecast_folder = './../../dataset_folder/Monash_forecasting_archive/archive/'
regression_folder = './../../dataset_folder/Monash_extrinsic_regression_archive/archive/'

##### forecast archive
#####
datasets = os.listdir(forecast_folder)
# random.shuffle(datasets)

default_start = datetime.datetime(1998, 1, 1)
frequency = {'4_seconds': '4S', 'minutely': 'T', '10_minutes': '10T', 'half_hourly': '0.5H', 'hourly':'H', 'daily':'D', 'weekly':'W', 'monthly':'MS', 'quarterly':'QS', 'None':None}

file = h5py.File("./../../dataset_folder/real_data.hdf5", "w")

## create group 
grp = file.create_group('monash_forecast')

for dataset in datasets:
    data = convert_tsf_to_dataframe(forecast_folder + dataset)
    dataset = os.path.splitext(dataset)[0]

    if data[1] == None:
        data[1] = 'None'

    data = {'df':data[0],'freq':data[1],'forecast_horizon':data[2],'missing_values':data[3],'equal_length':data[4]}
    df = data['df']

    ## create dataset group
    dset_group = grp.create_group(dataset)
    dset_group.attrs['frequency'] = data['freq']

    ## create dataset
    print('current dataset: ', dataset)
    data_list = df['series_value'].tolist()
    [dset_group.create_dataset(dataset+str(i), data=data_list[i], dtype=float) for i in range(len(data_list))]

file.close()

##### regression archive
#####

datasets = os.listdir(regression_folder)

trainfiles = [os.path.join(regression_folder,dataset)+'/'+dataset+'_TRAIN.ts' for dataset in datasets]
testfiles = [os.path.join(regression_folder,dataset)+'/'+dataset+'_TEST.ts' for dataset in datasets]


f = h5py.File("./../../dataset_folder/real_data.hdf5", "a")


## create group 
grp = f.create_group('monash_regression_test')

for file, dataset in zip(testfiles,datasets):
    print('######## ' + dataset)
    data = load_from_tsfile_to_dataframe(file)

    columns = data[0].columns

    data_list = []

    for i in range(len(data[0])):
        data_by_idx = data[0].iloc[i]
        data_list += [data_by_idx[val].tolist() for val in columns]

    ## create dataset group
    dset_group = grp.create_group(dataset)

    ## create dataset
    print('current dataset: ', dataset)
    [dset_group.create_dataset(dataset+str(i), data=data_list[i], dtype=float) for i in range(len(data[0]) * len(columns))]

f.close()
