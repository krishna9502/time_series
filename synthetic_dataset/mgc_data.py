import numpy as np
import pandas as pd

## class which includes methods to generate each mgc data
class MGC_generator:
    def __init__(self):
        self.base_limits = [10**4,10**5,10**6,10**7]
        self.ts_model = ['Uniform', 'Monotonic', 'Erratic', 'Sparse', 'Seasonal', 'Cyclic']
        self.dates = pd.date_range(start='1/1/2016',end='31/12/2018')
        self.first_days_idx, self.days_in_months = self.get_first_days()

    def pick_random_mgc(self):
        mgc_list = np.arange(11,51).reshape((4,10))[:,0:6].ravel()
        mgc = np.random.RandState().choice(mgc_list)
        return mgc

    def get_first_days(self):
        data = {'dates': self.dates}
        df = pd.DataFrame(data)
        first_days_idx = df.loc[df['dates'].dt.day == 1].index.to_list()
        days_in_months = df['dates'].dt.days_in_month.to_list()
        days_in_months = [days_in_months[i] for i in first_days_idx]
        return first_days_idx, days_in_months

    def calc_ol(self,data,limits):
        ol= []
        ol_num = []
        for limit in limits:
            ol_ = np.array(data) - np.array(limit)
            ol_ = (ol_>0) * ol_
            ol_num_ = np.sum(ol_>0)
            ol.append(ol_.mean())
            ol_num.append(ol_num_)
        return ol, ol_num

    def calc_gap(self,data,limits):
        gap= []
        for limit in limits:
            gap_ = np.array(limit) - np.array(data)
            gap_ = (gap_>0) * gap_
            gap.append(gap_.mean())
        return gap

    def generate_mgc_data(self,mgc):
        if mgc%10 == 1:
            ts_data = self.generate_uniform_data(mgc)
        elif mgc%10 == 2:
            ts_data = self.generate_monotonic_data(mgc)
        elif mgc%10 == 3:
            ts_data = self.generate_erratic_data(mgc)
        elif mgc%10 == 4:
            ts_data = self.generate_sparse_data(mgc)
        elif mgc%10 == 5:
            ts_data = self.generate_seasonal_data(mgc)
        elif mgc%10 == 6:
            ts_data = self.generate_cyclic_data(mgc)

        ## limit sampling
        ts_data_blocks = np.array(ts_data[-360:]).reshape((12,-1))

        limits = []
        ol_list = []
        gap_list = []
        olnum_list = []

        for block in ts_data_blocks:
            if (block.max()-block.min())/100 == 0:
                limit_block = np.ones(100) * self.base_limit/100
            else:
                limit_block = np.arange(block.min(),block.max(),(block.max()-block.min())/100)
            limits.append(limit_block)
            ol_list.append(self.calc_ol(block,limit_block)[0])
            gap_list.append(self.calc_gap(block,limit_block))
            olnum_list.append(self.calc_ol(block,limit_block)[1])

        return mgc, ts_data, limits, ol_list, gap_list, olnum_list  #list

    ## 1Uniform, choose: {anchor_value_center = (0.4,0.6), anchor_values_scale = 0.1, noise_factor = (0.1,0.5)} * base_limit
    def generate_uniform_data(self, mgc):
        base_limit = self.base_limits[mgc//10 - 1]
        self.base_limit = base_limit
        num_days_months = self.days_in_months
        ## create random anchor values at the beginning of each month
        anchor_value_center = np.random.RandomState().uniform(low = 0.4 * base_limit, high = 0.6 * base_limit)
        anchor_values = np.random.RandomState().normal(loc=anchor_value_center, scale=0.1*base_limit, size=len(self.first_days_idx)+1).tolist() # (n+1) points
        ## create vectorized points for entire data
        prev_points = np.repeat(np.array(anchor_values[:-1]),repeats = num_days_months)
        next_points = np.repeat(np.array(anchor_values[1:]),repeats = num_days_months)
        iterations_array = []
        [iterations_array.extend(np.arange(x).tolist()) for x in num_days_months]
        num_days_months_array = np.repeat(np.array(num_days_months),repeats= num_days_months)
        ## find the linear interp points between anchor points
        interp_values = prev_points + iterations_array * (next_points - prev_points)/num_days_months_array
        ## noise level around the interp points
        noise_factor1= np.random.RandomState().uniform(0.1,0.5, size = len(interp_values))
        noise_factor2= np.random.RandomState().uniform(0.1,0.5, size = len(interp_values))
        random_interp_values = interp_values + np.random.RandomState().randint(low= -1*base_limit*noise_factor1, high= base_limit*noise_factor2, size= len(interp_values))
        random_interp_values = np.round_(random_interp_values.clip(0,None)).tolist()

        return random_interp_values

    ## 2Monotonic, choose: {anchor_value_center = 0.5, anchor_values_scale = 0.1, noise_factor = (0.1,0.7)} * base_limit
    def generate_monotonic_data(self, mgc):
        base_limit = self.base_limits[mgc//10 - 1]
        self.base_limit = base_limit
        num_days_months = self.days_in_months
        ## create random anchor values at the beginning of each month
        anchor_value_center = 0.5 * base_limit
        sort_order = 1 if np.random.RandomState().random() < 0.5 else -1
        anchor_values = sort_order * np.sort(sort_order * np.random.RandomState().normal(loc=anchor_value_center, scale=0.1*base_limit, size=len(self.first_days_idx)+1)) # (n+1) points
        anchor_values = anchor_values.tolist()
        ## create vectorized points for entire data
        prev_points = np.repeat(np.array(anchor_values[:-1]),repeats = num_days_months)
        next_points = np.repeat(np.array(anchor_values[1:]),repeats = num_days_months)
        iterations_array = []
        [iterations_array.extend(np.arange(x).tolist()) for x in num_days_months]
        num_days_months_array = np.repeat(np.array(num_days_months),repeats= num_days_months)
        ## find the linear interp points between anchor points
        interp_values = prev_points + iterations_array * (next_points - prev_points)/num_days_months_array
        ## noise level around the interp points
        noise_factor1= np.random.RandomState().uniform(0.1,0.7, size = len(interp_values))
        noise_factor2= np.random.RandomState().uniform(0.1,0.7, size = len(interp_values))
        random_interp_values = interp_values + np.random.RandomState().randint(low= -1*base_limit*noise_factor1, high= base_limit*noise_factor2, size= len(interp_values))
        random_interp_values = np.round_(random_interp_values.clip(0,None)).tolist()

        return random_interp_values

    ## 3Erratic, choose: {anchor_value_center = 0.5, anchor_values_scale = 0.1, noise_factor = (0.5,1.0)} * base_limit
    def generate_erratic_data(self, mgc):
        base_limit = self.base_limits[mgc//10 - 1]
        self.base_limit = base_limit
        num_days_months = self.days_in_months
        ## create random anchor values at the beginning of each month
        anchor_value_center = 0.5 * base_limit
        anchor_values = np.random.RandomState().normal(loc=anchor_value_center, scale=0.1*base_limit, size=len(self.first_days_idx)+1).tolist() # (n+1) points
        ## create vectorized points for entire data
        prev_points = np.repeat(np.array(anchor_values[:-1]),repeats = num_days_months)
        next_points = np.repeat(np.array(anchor_values[1:]),repeats = num_days_months)
        iterations_array = []
        [iterations_array.extend(np.arange(x).tolist()) for x in num_days_months]
        num_days_months_array = np.repeat(np.array(num_days_months),repeats= num_days_months)
        ## find the linear interp points between anchor points
        interp_values = prev_points + iterations_array * (next_points - prev_points)/num_days_months_array
        ## noise level around the interp points
        noise_factor1= np.random.RandomState().uniform(0.5,1.0, size = len(interp_values))
        noise_factor2= np.random.RandomState().uniform(0.5,1.0, size = len(interp_values))
        random_interp_values = interp_values + np.random.RandomState().randint(low= -1*base_limit*noise_factor1, high= base_limit*noise_factor2, size= len(interp_values))
        random_interp_values = np.round_(random_interp_values.clip(0,None)).tolist()

        return random_interp_values

    ## 4Sparse, choose: {anchor_value_center = 0.5, anchor_values_scale = 0.1, #p_empty_space = (0.5,0.8), noise_factor = (0.5,1.0)} * base_limit
    def generate_sparse_data(self, mgc):
        base_limit = self.base_limits[mgc//10 - 1]
        self.base_limit = base_limit
        num_days_months = self.days_in_months
        ## create random anchor values at the beginning of each month
        anchor_value_center = 0.5 * base_limit
        anchor_values = np.random.RandomState().normal(loc=anchor_value_center, scale=0.1*base_limit, size=len(self.first_days_idx)+1).tolist() # (n+1) points
        ## create vectorized points for entire data
        prev_points = np.repeat(np.array(anchor_values[:-1]),repeats = num_days_months)
        next_points = np.repeat(np.array(anchor_values[1:]),repeats = num_days_months)
        iterations_array = []
        [iterations_array.extend(np.arange(x).tolist()) for x in num_days_months]
        num_days_months_array = np.repeat(np.array(num_days_months),repeats= num_days_months)
        ## find the linear interp points between anchor points
        interp_values = prev_points + iterations_array * (next_points - prev_points)/num_days_months_array
        ## create mask on interp_values
        p = np.random.RandomState().uniform(0.5,0.8,size=2)
        mask_values1 = np.random.RandomState().choice(np.array([0,1]), size= len(interp_values)//2, p= [p[0],1-p[0]])
        mask_values2 = np.random.RandomState().choice(np.array([0,1]), size= len(interp_values)//2, p= [p[1],1-p[1]])
        num_ones1 = np.sum(mask_values1)
        num_ones2 = np.sum(mask_values2)
        mask_values1 = np.zeros(len(mask_values1)-num_ones1)
        insert_idx1 = np.random.RandomState().randint(len(mask_values1))
        mask_values1 = np.insert(mask_values1, insert_idx1, np.ones(num_ones1))
        mask_values2 = np.zeros(len(mask_values2)-num_ones2)
        insert_idx2 = np.random.RandomState().randint(len(mask_values2))
        mask_values2 = np.insert(mask_values2, insert_idx2, np.ones(num_ones2))
        mask_values = np.concatenate((mask_values1, mask_values2))
        ## noise level around the interp points
        noise_factor1= np.random.RandomState().uniform(0.5,1.0, size = len(interp_values))
        noise_factor2= np.random.RandomState().uniform(0.5,1.0, size = len(interp_values))
        random_interp_values = interp_values + np.random.RandomState().randint(low= -1*base_limit*noise_factor1, high= base_limit*noise_factor2, size= len(interp_values))
        random_interp_values = np.round_(random_interp_values.clip(0,None)).tolist()
        random_interp_values = random_interp_values * mask_values
        random_interp_values = random_interp_values.tolist()

        return random_interp_values

    ## 5Seasonal, choose: {anchor_value_center = (0.4,0.6), anchor_values_scale = 0.1, noise_factor = (0.1,0.5)} * base_limit
    def generate_seasonal_data(self, mgc):
        base_limit = self.base_limits[mgc//10 - 1]
        self.base_limit = base_limit
        anchor_values = None
        random_interp_values3 = []
        for year in range(3):
            num_days_months = self.days_in_months[year*12:(year+1)*12]
            ## create random anchor values at the beginnin of each month
            anchor_value_center = np.random.RandomState().uniform(low = 0.4 * base_limit, high = 0.6 * base_limit)
            if year == 0:
                anchor_values = np.random.RandomState().normal(loc=anchor_value_center, scale=0.1*base_limit, size=len(self.first_days_idx)//3+1).tolist() # (n+1) points
            ## create vectorized poitns for entire data
            prev_points = np.repeat(np.array(anchor_values[:-1]),repeats = num_days_months)
            next_points = np.repeat(np.array(anchor_values[1:]),repeats = num_days_months)
            iterations_array = []
            [iterations_array.extend(np.arange(x).tolist()) for x in num_days_months]
            num_days_months_array = np.repeat(np.array(num_days_months),repeats= num_days_months)
            ## find the linear interp points between anchor points
            interp_values = prev_points + iterations_array * (next_points - prev_points)/num_days_months_array
            ## noise level around the interp points
            noise_factor1= np.random.RandomState().uniform(0.1,0.5, size = len(interp_values))
            noise_factor2= np.random.RandomState().uniform(0.1,0.5, size = len(interp_values))
            random_interp_values = interp_values + np.random.RandomState().randint(low= -1*base_limit*noise_factor1, high= base_limit*noise_factor2, size= len(interp_values))
            random_interp_values = np.round_(random_interp_values.clip(0,None)).tolist()
            random_interp_values3 += random_interp_values

        return random_interp_values3

    ## 6Cyclic, choose: {#period =(4,11), anchor_value_center = (0.4,0.6), anchor_values_scale = 0.1, noise_factor = (0.1,0.5)} * base_limit
    def generate_cyclic_data(self, mgc):
        period = np.random.RandomState().randint(4,11)
        base_limit = self.base_limits[mgc//10 - 1]
        self.base_limit = base_limit
        anchor_values = None
        random_interp_valuesN = []
        iteration = 0
        while True: 
            end = (iteration+1)*period
            if end >= len(self.days_in_months):
                num_days_months = self.days_in_months[iteration*period:]
                anchor_values = anchor_values[:len(num_days_months)+1]
            else:
                num_days_months = self.days_in_months[iteration*period:end]
            ## create random anchor values at the beginnin of each month
            anchor_value_center = np.random.RandomState().uniform(low = 0.4 * base_limit, high = 0.6 * base_limit)
            if iteration==0:
                anchor_values = np.random.RandomState().normal(loc=anchor_value_center, scale=0.1*base_limit, size=period+1).tolist() # (n+1) points
            ## create vectorized poitns for entire data
            prev_points = np.repeat(np.array(anchor_values[:-1]),repeats = num_days_months)
            next_points = np.repeat(np.array(anchor_values[1:]),repeats = num_days_months)
            iterations_array = []
            [iterations_array.extend(np.arange(x).tolist()) for x in num_days_months]
            num_days_months_array = np.repeat(np.array(num_days_months),repeats= num_days_months)
            ## find the linear interp points between anchor points
            interp_values = prev_points + iterations_array * (next_points - prev_points)/num_days_months_array
            ## noise level around the interp points
            noise_factor1= np.random.RandomState().uniform(0.1,0.5, size = len(interp_values))
            noise_factor2= np.random.RandomState().uniform(0.1,0.5, size = len(interp_values))
            random_interp_values = interp_values + np.random.RandomState().randint(low= -1*base_limit*noise_factor1, high= base_limit*noise_factor2, size= len(interp_values))
            random_interp_values = np.round_(random_interp_values.clip(0,None)).tolist()
            random_interp_valuesN += random_interp_values
            iteration+=1
            if len(random_interp_valuesN) >= np.sum(self.days_in_months):
                random_interp_valuesN = random_interp_valuesN[:np.sum(self.days_in_months)]
                break

            return random_interp_valuesN






