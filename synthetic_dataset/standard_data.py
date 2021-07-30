import numpy as np

class Standard_generator():
    def __init__(self, data_size=1000):
        self.data_size = data_size
        self.x_vals  = np.arange(0,data_size)

    # def generate_trend_data():
    # x_anchors = np.arange()

    def generate_seasonal_data(self,num_frequencies=3):
        T_list = np.random.choice(np.arange(5,51), size=num_frequencies, replace=False)
        data = np.zeros(self.data_size)
        for T in T_list:
            data += np.sin(2*np.pi*self.x_vals/T)
        return data 

    def generate_noise_data(self,var=1.0):
        y_t_1 = 0
        data = []
        for i in range(self.data_size):
            z_t = np.random.normal(0,var)
            y_t = y_t_1 + z_t
            data.append(y_t)
            y_t_1 = y_t
        return data
