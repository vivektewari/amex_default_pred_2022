import pandas as pd,numpy as np
import random,json
import torch
import time
from multiprocessing import Pool, cpu_count
from input_diagnostics.iv import IV
from input_diagnostics.common import *
import importlib

random.seed(24)
a=IV(verbose=False)
a.load(config.output_loc+'eda/feature_importance/')
a.excludeList=['customer_ID','target']
with open(config.weight_loc +"binned_vlist", "r") as fp:var_list =json.load(fp)

def get_pred2(train):

    converted = a.convertToWoe(train)
    converted['customer_ID'] = train['customer_ID']
    converted['target'] = train['target']
    #converted['Unnamed: 0']=-0.04137
    return converted




def get_pred(model,var_list,output_loc,loc,from_row, to_row,tranformation=None):
    key = "customer_ID"
    def breakinUsers(r):
        r = r.drop(key, axis=1)
        return r.to_numpy()
    #train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')

    train = pd.read_csv(loc, skiprows=range(1, from_row + 1),
                        nrows=to_row - from_row, header=0)
    if tranformation is not None:train=tranformation(train)
    # train = train.set_index(key).select_dtypes(
    #     exclude=['object', 'O']).reset_index()  # todo check for better replacement
    if 'target' in var_list: var_list.remove('target')
    train=train[var_list]
    train = train.replace([np.inf, -np.inf, np.nan, np.inf], 0.00)

    if train.shape[0] > 1:
        output_dict = {'customer_ID': [], 'prediction': []}
        group = train.groupby('customer_ID').apply(breakinUsers).to_dict()
        max_seq = 13
        temp = []
        for ke in group.keys():
            output_dict['customer_ID'].append(ke)
            tuparray = torch.from_numpy(group[ke].astype(np.float))
            seq_len = tuparray.shape[0]
            # tuparray = np.array(tup)

            if seq_len >= max_seq:
                outs = tuparray[-max_seq:, :]
            else:
                outs = torch.cat((torch.zeros((max_seq-seq_len,tuparray.shape[1])),tuparray[-seq_len:,:]))

            temp.append(outs.to(torch.float32))

        outs = torch.stack(temp, dim=0)
        output_dict['prediction'] = model(outs).detach().tolist()


        pd.DataFrame(output_dict).to_csv(output_loc, index=False, mode='a', header=False)
        print('appended {}-{}'.format(from_row,to_row))

def get_inference(model,loc="",output_loc="",varlist="",keep_actual=True,pred_rows=10000,transformation=None):

    start_time=time.time()
    key='customer_ID'
    df = pd.read_csv(loc, usecols=[key])
    max_row = df.shape[0]
    rows = df[[key]].drop_duplicates(subset=['customer_ID'], keep='last').index.to_list()
    df,df1 = 0,0 # for memmory efficiency
    loop = 1
    from_row = 0
    #max_row = rows[-1] + 1

    pd.DataFrame(columns=['customer_ID', 'prediction']).to_csv(output_loc, index=False)
    #pool=Pool(cpu_count())


    while from_row <= max_row:
        if from_row == max_row:
            break
        elif loop * pred_rows <= len(rows):
            to_row = rows[loop * pred_rows] + 1
        else:
            to_row = max_row

            #pool.apply_async(get_pred, args=(model,varlist_loc,output_loc,loc,from_row, to_row,))
        get_pred(model,varlist,output_loc,loc,from_row, to_row,tranformation=transformation)
        loop += 1
        from_row = to_row
    #pool.close()
    #pool.join()


    df = pd.read_csv(output_loc)
    if keep_actual:
        df_target=pd.read_csv(loc,usecols=['customer_ID','target']).groupby('customer_ID').tail(1).set_index('customer_ID')

        df = df.join(df_target,on='customer_ID')

    df.to_csv(output_loc, index=False)
    print('rows written {} time taken {}'.format(df.shape[0],time.time()-start_time))
    return df
