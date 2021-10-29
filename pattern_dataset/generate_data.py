import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from statsmodels.tsa.seasonal import seasonal_decompose

def get_anchor_xvalues(x_values,num_anchors):
    anchor_xvalues = []
    anchor_xvalues.append(x_values[0])

    anchor_pos = x_values[0]

    for i in range(num_anchors-2):
        anchor_num = i
        anchor_pos = (i+1) * x_values[-1]/(num_anchors-1)
        anchor_xvalues.append(anchor_pos)

    anchor_xvalues.append(x_values[-1])

    return anchor_xvalues

def generate_noise_weekly(x_values, noise_factor):
    patterns = np.array([[1,1,-1,-1,-1,-1,1],\
                         [-1,-1,-1,1,1,-1,-1],\
                         [-1,1,1,-1,-1,1,-1]])
    data = []
    for i in range(157):
        week_rands = np.random.randint(0,7,7)/10
        np.random.shuffle(patterns)
        pattern = patterns[0]

        noise = pattern * week_rands + np.abs(np.random.normal([0,0,0,0,0,0,0],[0,0,0,0,0,0.4,0.4]))
        noise = noise * np.abs(np.random.normal(1,0.3))
        data.append(noise)
    data = np.array(data).ravel()[0:1096]

    return data

def get_first_days_months(dates):
    data = {'dates':dates}
    df = pd.DataFrame(data)
    first_days_idx = df.loc[df['dates'].dt.day==1].index.to_list()
    days_in_months = df['dates'].dt.days_in_month.to_list()
    days_in_months = [days_in_months[i] for i in first_days_idx]
    return first_days_idx, days_in_months

def get_first_days_weeks(dates):
    length = len(dates)
    first_days_idx = np.arange(length)%7
    first_days_idx = np.where(first_days_idx == 0)[0]
    return first_days_idx

txn_data = []
dates = pd.date_range(start='1/1/2016',end='31/12/2018')
first_days_idx_month, days_in_months = get_first_days_months(dates)
first_days_idx_week = get_first_days_weeks(dates)
anchor_limit = 10**6

x_values = np.arange(len(dates))
num_anchors = 4

df = pd.DataFrame(columns=['merchant','date','time_idx','value'])

for i in range(1000):
    anchor_xvalues = get_anchor_xvalues(x_values,num_anchors)
    anchor_yvalues = np.random.randint(anchor_limit*0.3,anchor_limit*0.7,num_anchors)

    y_values = np.interp(x_values,anchor_xvalues, anchor_yvalues)

    noise = generate_noise_weekly(x_values,100000)

    data = noise

    # monthly spikes
    vals = []
    for j in range(len(data)//30):
        vals_ = np.abs(np.random.normal(0.0,0.1,size=30))
        vals.append(np.sort(vals_)[::-1])
    vals = np.array(vals).ravel()
    vals = np.pad(vals,(0,len(data)-len(vals)),'constant')

    data = data + vals

    ## outliers
    rand_indices = np.random.choice(1095,size=5,replace=False)

    for item in rand_indices:
        data[item] +=np.abs(np.random.normal(0,3))

    noise_factor = 100000
    data *= noise_factor

    y_values = y_values + data

    df_ = pd.DataFrame(data={'merchant':['M'+str(i)]*len(dates), 'date': dates, 'time_idx':x_values, 'value': y_values})

    df = pd.concat([df,df_], ignore_index=True)

    print('iteration-'+str(i))

df.to_csv('pattern_data.csv',index=False)
 





































