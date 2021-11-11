import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
        noise = noise * np.abs(np.random.normal(1,0.3)) * noise_factor
        data.append(noise)
    data = np.array(data).ravel()[0:1096]

    return data



### main code
dates = pd.date_range(start='1/1/2016',end='31/12/2018')
anchor_limit = 10**6

first_days_idx_month, days_in_months = get_first_days_months(dates)
first_days_idx_week = get_first_days_weeks(dates)

anchor_xvalues = [0,366,731,1065]
x_values = np.arange(len(dates))


# ## 1.trend + pattern noise
df = pd.DataFrame(columns=['merchant','date','time_idx','value'])

# fig,ax = plt.subplots()

for idx in range(1000):
    merchant = 'M' + str(idx) + '_1'

    anchor_yvalues = np.random.randint(anchor_limit*0.3,anchor_limit*0.7,4)
    y_values = np.interp(x_values,anchor_xvalues, anchor_yvalues)
    noise = generate_noise_weekly(x_values,10**5)

    y_values += noise

    # ax.cla()
    # ax.plot(y_values)
    # ax.hlines([0,10**6],0,1095,color='r')
    # fig.canvas.draw()
    # plt.pause(1)

    df_ = pd.DataFrame(data={'merchant':[merchant]*len(dates), 'date': dates, 'time_idx':x_values, 'value': y_values})
    df = pd.concat([df,df_], ignore_index=True)

    print('iteration-'+str(idx))

df.to_csv('pattern_data1.csv',index=False)


## 2.seasonal trend + pattern noise
df = pd.DataFrame(columns=['merchant','date','time_idx','value'])

# fig,ax = plt.subplots()

for idx in range(1000):
    merchant = 'M' + str(idx) + '_2'

    anchor_yvalues = np.random.randint(anchor_limit*0.3,anchor_limit*0.7,4)
    y_values = np.interp(x_values,anchor_xvalues, anchor_yvalues)

    ## sinusoid
    amplitude = 10**5
    y_sinusoid = amplitude * np.sin(2*np.pi*x_values/np.random.randint(180,500))
    y_values += y_sinusoid 

    noise = generate_noise_weekly(x_values,10**5)
    y_values += noise

    # ax.cla()
    # ax.plot(y_values)
    # ax.hlines([0,10**6],0,1095,color='r')
    # fig.canvas.draw()
    # plt.pause(1)

    df_ = pd.DataFrame(data={'merchant':[merchant]*len(dates), 'date': dates, 'time_idx':x_values, 'value': y_values})
    df = pd.concat([df,df_], ignore_index=True)

    print('iteration-'+str(idx))

df.to_csv('pattern_data2.csv',index=False)


## 3.trend + seasonal pattern noise
df = pd.DataFrame(columns=['merchant','date','time_idx','value'])

# fig,ax = plt.subplots()

for idx in range(1000):

    merchant = 'M' + str(idx) + '_3'

    anchor_yvalues = np.random.randint(anchor_limit*0.3,anchor_limit*0.7,4)
    y_values = np.interp(x_values,anchor_xvalues, anchor_yvalues)
    noise = generate_noise_weekly(x_values,10**5)
    y_values += noise

    ## seasonal spikes
    start_spike = np.random.randint(90)
    gap_spike = np.random.randint(120,300)
    spike_xvalues = []
    spike = start_spike
    while spike < len(x_values):
        spike_xvalues.append(spike)
        spike += gap_spike

    mask = np.zeros(len(x_values))
    mask[spike_xvalues] = 1

    spike_yvalues = mask * np.random.uniform(3*10**5,4*10**5,len(x_values))

    y_values += spike_yvalues

    # ax.cla()
    # ax.plot(y_values)
    # ax.hlines([0,10**6],0,1095,color='r')
    # fig.canvas.draw()
    # plt.pause(1)

    df_ = pd.DataFrame(data={'merchant':[merchant]*len(dates), 'date': dates, 'time_idx':x_values, 'value': y_values})
    df = pd.concat([df,df_], ignore_index=True)

    print('iteration-'+str(idx))

df.to_csv('pattern_data3.csv',index=False)


## 4.seasonal trend + seasonal pattern noise
df = pd.DataFrame(columns=['merchant','date','time_idx','value'])

# fig,ax = plt.subplots()

for idx in range(1000):
    merchant = 'M' + str(idx) + '_4'

    anchor_yvalues = np.random.randint(anchor_limit*0.3,anchor_limit*0.7,4)
    y_values = np.interp(x_values,anchor_xvalues, anchor_yvalues)
    noise = generate_noise_weekly(x_values,10**5)
    y_values += noise

    ## sinusoid
    amplitude = 10**5
    y_sinusoid = amplitude * np.sin(2*np.pi*x_values/np.random.randint(180,500))
    y_values += y_sinusoid 

    ## seasonal spikes
    start_spike = np.random.randint(90)
    gap_spike = np.random.randint(120,300)
    spike_xvalues = []
    spike = start_spike
    while spike < len(x_values):
        spike_xvalues.append(spike)
        spike += gap_spike

    mask = np.zeros(len(x_values))
    mask[spike_xvalues] = 1

    spike_yvalues = mask * np.random.uniform(3*10**5,4*10**5,len(x_values))

    y_values += spike_yvalues

    # ax.cla()
    # ax.plot(y_values)
    # ax.hlines([0,10**6],0,1095,color='r')
    # fig.canvas.draw()
    # plt.pause(1)

    df_ = pd.DataFrame(data={'merchant':[merchant]*len(dates), 'date': dates, 'time_idx':x_values, 'value': y_values})
    df = pd.concat([df,df_], ignore_index=True)

    print('iteration-'+str(idx))

df.to_csv('pattern_data4.csv',index=False)

## 5.mixture
df = pd.DataFrame(columns=['merchant','date','time_idx','value'])

# fig,ax = plt.subplots()

for idx in range(1000):

    merchant = 'M' + str(idx) + '_5'

    anchor_yvalues = np.random.randint(anchor_limit*0.3,anchor_limit*0.7,4)
    y_values = np.interp(x_values,anchor_xvalues, anchor_yvalues)
    noise = generate_noise_weekly(x_values,10**5)
    y_values += noise

    ## sinusoid
    if random.choice([True, False]):
        amplitude = 10**5
        y_sinusoid = amplitude * np.sin(2*np.pi*x_values/np.random.randint(180,500))
        y_values += y_sinusoid 

    ## seasonal spikes
    if random.choice([True, False]):

        start_spike = np.random.randint(90)
        gap_spike = np.random.randint(120,300)
        spike_xvalues = []
        spike = start_spike
        while spike < len(x_values):
            spike_xvalues.append(spike)
            spike += gap_spike

        mask = np.zeros(len(x_values))
        mask[spike_xvalues] = 1

        spike_yvalues = mask * np.random.uniform(3*10**5,4*10**5,len(x_values))

        y_values += spike_yvalues

    # ax.cla()
    # ax.plot(y_values)
    # ax.hlines([0,10**6],0,1095,color='r')
    # fig.canvas.draw()
    # plt.pause(1)

    df_ = pd.DataFrame(data={'merchant':[merchant]*len(dates), 'date': dates, 'time_idx':x_values, 'value': y_values})
    df = pd.concat([df,df_], ignore_index=True)

    print('iteration-'+str(idx))

df.to_csv('pattern_data5.csv',index=False)
