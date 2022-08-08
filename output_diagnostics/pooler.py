from multiprocessing import Pool,cpu_count
import torch
import pandas as pd
import numpy as np
pool=Pool(cpu_count())
def get_pred(model,varlist_loc,output_loc,loc,from_row, to_row):

    def breakinUsers(r):
        r = r.drop(varlist_loc, axis=1)
        return r.to_numpy()
    #train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')
    key="customer_ID"
    train = pd.read_csv(loc, skiprows=range(1, from_row + 1),
                        nrows=to_row - from_row, header=0)
    train = train.set_index(key).select_dtypes(
        exclude=['object', 'O']).reset_index()  # todo check for better replacement
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
                outs = torch.cat((tuparray, torch.ones((max_seq - seq_len, tuparray.shape[1]))))

            temp.append(outs.to(torch.float32))

        outs = torch.stack(temp, dim=0)
        output_dict['prediction'] = model(outs).detach().tolist()

        pd.DataFrame(output_dict).to_csv(output_loc, index=False, mode='a', header=False)
        print('appended {}-{}'.format(from_row,to_row))