import pandas as pd

from common import *
import random
file_name='train_data.csv'
dev_perc=0.60 # keep only 60 % in dev instad of 80 so the feature importance counld be calculated
train =pd.read_csv(config.data_loc+'from_kaggle/'+file_name,usecols=['customer_ID'])
unique_list=list(train['customer_ID'].unique())
unique_count=len(unique_list)
dev_list=random.sample(unique_list,int(unique_count*dev_perc))
val_list=set(unique_list).difference(set(dev_list))


#iteratively making dev and hold out sample as data is too large to open in one go
pred_rows=1000000
max_row=train.shape[0]
train=pd.read_csv(config.data_loc+'from_kaggle/'+file_name,nrows=0,header=0)
dummy=pd.DataFrame(columns=list(train.columns)+['target'])
dummy.to_csv(config.data_loc+'data_created/dev.csv',index=False)
dummy.to_csv(config.data_loc+'data_created/hold_out.csv',index=False)
train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')
from_row = 0
to_row = min(from_row + pred_rows, max_row)
while from_row <= max_row:
    train = pd.read_csv(config.data_loc+'from_kaggle/'+file_name, skiprows=range(1, from_row+1), nrows=pred_rows, header=0)
    train=train.join(train_label,on=['customer_ID'])
    train[train['customer_ID'].isin(dev_list)].to_csv(config.data_loc+'data_created/dev.csv',mode='a',header=False,index=False)
    train[train['customer_ID'].isin(val_list)].to_csv(config.data_loc+'data_created/hold_out.csv',mode='a',header=False,index=False)
    print('completed for {}'.format(to_row))
    from_row = to_row
    to_row=to_row+pred_rows

#check
df_dev = pd.read_csv(config.data_loc+'data_created/dev.csv',usecols=['customer_ID'])

df_hold = pd.read_csv(config.data_loc+'data_created/hold_out.csv',usecols=['customer_ID'])
print('written files total are {},{},{},{}'.format(max_row,df_dev.shape[0]+df_hold.shape[0],df_dev.shape[0],df_hold.shape[0]))