import pandas as pd
import numpy as np
pd.__version__
import matplotlib.pyplot as plt
from datetime import datetime
# %matplotlib inline

Train = pd.read_csv('../../Data Science/historical_data.csv')
Train.dtypes
Train.isna().sum()

Train = Train[~Train.actual_delivery_time.isna()]

Train.isna().any()
Train.created_at[0].seconds()
Train.set_index('store_id').loc[Train[Train.store_primary_category.isna()].store_id.iloc[316]].store_primary_category
Train[Train.store_primary_category.isna()].store_id.iloc[100]
Train.store_id.describe()
Train[Train.store_primary_category.isna()]
Train.iloc[1]
Train.describe()
Train.head()

Train.columns
Train.dtypes
Train.created_at.astype("datetime64[s]")[1]  
Train.created_at[1].astype("datetime64[s]")[1]  

Train.iloc[1]
(Train.actual_delivery_time.head().astype("datetime64[s]")[1] - Train.created_at.astype("datetime64[s]")[1]).seconds

Train.estimated_order_place_duration[1] + Train.estimated_store_to_consumer_driving_duration[1]

Train.actual_delivery_time.astype("datetime64[s]")[1]  
Train['actual_delivery_time'].astype("datetime64[s]")

foo = datetime.datetime.now()
type(foo)


datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
fdk
foo = datetime.strptime('2018-07-29 21:49:25', '%Y-%m-%d %H:%M:%S')
foo.weekday()

Train.head()

test = pd.read_json('../../Data Science/data_to_predict.json', lines = True)
test.isnull().any()
test.head()
