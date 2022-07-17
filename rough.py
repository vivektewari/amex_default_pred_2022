import yaml
from types import SimpleNamespace
import pandas as pd
import numpy as np

with open('config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
a=pd.read_csv(config.data_loc+'from_kaggle/'+'train_data.csv',usecols=['customer_ID'])['customer_ID'].nunique()
b=pd.read_csv(config.data_loc+'data_created/'+'dev.csv',usecols=['customer_ID'])['customer_ID'].nunique()
c=pd.read_csv(config.data_loc+'data_created/'+'hold_out.csv',usecols=['customer_ID'])['customer_ID'].nunique()
d=0#pd.read_csv(config.data_loc+'from_kaggle/'+'test_data.csv',usecols=['customer_ID']).shape[0]
print(a,b,c,b+c,d)