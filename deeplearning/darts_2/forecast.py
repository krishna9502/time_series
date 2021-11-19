import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import RNNModel, TransformerModel, NBEATSModel
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler


torch.manual_seed(1); np.random.seed(1)  # for reproducibility

data = pd.read_csv('pattern_data5.csv')

merchants = data.merchant.unique()

print(data)




# series_list = []

# i=0
# for merchant in merchants:
#     df = data.loc[data.merchant==merchant]
#     ts = TimeSeries.from_dataframe(df, 'date', 'value')
#     scaler = Scaler()
#     ts_scaled = scaler.fit_transform(ts)
#     series_list.append(ts_scaled)
#     print(merchant)
#     i+=1
#     if i > 10:
#         break

# model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10, random_state=0)
# model.fit(series_list, verbose=True)

# for i,series in enumerate(series_list):
#     pred = model.predict(n=200, series=series)
#     series.plot(label='actual')
#     pred.plot(label='forecast')
#     plt.legend()
#     plt.savefig('NBEATS-'+str(i)+'.png')
#     plt.clf()