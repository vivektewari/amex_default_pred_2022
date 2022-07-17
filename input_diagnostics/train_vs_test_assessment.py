import pandas as pd

from common import *
#anylyzing test and train dataset so can develop dev and holdout sample in same way.
# This will ensure consistency of clculated metrics of test and hold out sample
# Q1. is customer_id in test and train are mutually exclusive
# Q2. for any particular customer_id does all time records (13 or 18 month are present in both
#A2:train and test are same , for each customer it is a monthly data but day is not end of month day


# train =pd.read_csv(config.data_loc+'from_kaggle/train_data.csv',nrows=1000)
# test =pd.read_csv(config.data_loc+'from_kaggle/test_data.csv',nrows=1000)
train_c =pd.read_csv(config.data_loc+'from_kaggle/train_data.csv',usecols=['customer_ID'])
test_c =pd.read_csv(config.data_loc+'from_kaggle/train_data.csv',usecols=['customer_ID'])
intersection=set(train_c['customer_ID']).intersection(set(test_c['customer_ID']))
print(len(intersection))
c=0