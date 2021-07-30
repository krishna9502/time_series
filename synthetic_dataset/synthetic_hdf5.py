import numpy as np
import h5py
import random
import os
from tqdm import tqdm 

from mgc_data import MGC_generator
from standard_data import * 

# file = h5py.File("./../../dataset_folder/synthetic_data.hdf5", "w")


# ## limit datasets
# main_grp = file.create_group('limit_datasets')

# categories = {'uniform':1,'monotonic':2,'erratic':3,'sparse':4,'seasonal':5,'cyclic':6}
# base_limits = [10**4,10**5,10**6,10**7]

# mgc_generator = MGC_generator()

# for category in list(categories.keys()):
#     ## create group 
#     grp = main_grp.create_group(category)

#     for iteration in range(1000):

#         base_limit_idx = random.randint(0,3)
#         mgc = 10 * (base_limit_idx+1) + categories[category]

#         data = mgc_generator.generate_mgc_data(mgc=mgc)[1] # list
        
#         dataset = grp.create_dataset('M_'+f'{iteration:03d}'+'_'+str(mgc), data=data, dtype=float)
#         print('M_'+f'{iteration:03d}'+'_'+str(mgc) + ' created!')

# file.close()


## standard datasets

file = h5py.File("./../../dataset_folder/synthetic_data.hdf5", "a")

del file['standard_datasets']
main_grp = file.create_group('standard_datasets')

categories = {'seasonal':1, 'noise':2}

standard_generator = Standard_generator(data_size=1000)

for category in list(categories.keys()):
    ## create group
    grp = main_grp.create_group(category)

    for iteration in range(100):
        if category =='seasonal':
            data = standard_generator.generate_seasonal_data(num_frequencies=2)
            dataset = grp.create_dataset('data_'+f'{iteration:03d}'+'_'+str(categories[category]), data=data, dtype=float)
            print('data_'+f'{iteration:03d}'+'_'+str(categories[category]) + ' created!')

        if category =='noise':
            data = standard_generator.generate_noise_data(var=0.1)
            dataset = grp.create_dataset('data_'+f'{iteration:03d}'+'_'+str(categories[category]), data=data, dtype=float)
            print('data_'+f'{iteration:03d}'+'_'+str(categories[category]) + ' created!')

file.close()