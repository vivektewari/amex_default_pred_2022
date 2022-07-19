import pandas as pd,numpy as np
import random,json
import torch

random.seed(24)
def get_pred(model,train,varlist_loc,multi_class=True):
    #train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')
    if 'target' in train.columns:target=train['target'].to_numpy()
    else :target=np.zeros(train.shape[0])
    with open(varlist_loc, "r") as fp:var_list =json.load(fp)
    train=train.replace([np.inf,-np.inf,np.nan,np.inf],0.00)
    train=train[var_list]
    X_train= train.to_numpy()
    pred=model(X_train)
    if multi_class:pred=pred[:,1]
    return pred,target
def get_inference(model,loc="",output_loc="",varlist_loc="",keep_actual=True,pred_rows=100000):
    def breakinUsers(r):
        r = r.drop(varlist_loc, axis=1)
        return r.to_numpy()

    key='customer_ID'
    target='target'
    df = pd.read_csv(loc, usecols=[key])
    max_row = df.shape[0]
    df1=pd.read_csv(loc, nrows=2)
    target_available=False
    if target in df1.columns:target_available=True
    if target_available:
        target_dict = pd.read_csv(loc, usecols=[key, target], nrows=max_row).groupby('customer_ID').tail(1).set_index(key)[target]

    rows = df[[key]].drop_duplicates(subset=['customer_ID'], keep='last').index.to_list()
    df,df1 = 0,0 # for memmory efficiency
    group = pd.Series([])
    loop = 1
    from_row = 0
    max_row = rows[-1] + 1
    print(max_row)
    pd.DataFrame(columns=['customer_ID', 'prediction', 'target']).to_csv(output_loc, index=False)
    while from_row <= max_row:
        if from_row == max_row:
            break
        elif loop * pred_rows <= len(rows):
            to_row = rows[loop * pred_rows] + 1
        else:
            to_row = max_row
        train = pd.read_csv(loc, skiprows=range(1, from_row + 1),
                            nrows=to_row - from_row, header=0)
        train = train.set_index(key).select_dtypes(
            exclude=['object', 'O']).reset_index()  # todo check for better replacement
        train = train.replace([np.inf, -np.inf, np.nan, np.inf], 0.00)
        if train.shape[0] > 1:
            output_dict={'customer_ID':[],'prediction':[],'target':[]}
            group = train.groupby('customer_ID').apply(breakinUsers).to_dict()
            max_seq=13
            temp=[]
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

                if target_available:output_dict['target'].append(target_dict[ke])
                else :output_dict['target'].append(0)
            outs = torch.stack(temp, dim=0)
            output_dict['prediction']=model(outs).detach().tolist()


            pd.DataFrame(output_dict).to_csv(output_loc,index=False,mode='a',header=False)
        loop += 1
        print(from_row, to_row)
        from_row = to_row
    df = pd.read_csv(output_loc)
    if not keep_actual:
        df = df[['customer_ID', 'prediction']]
    df.to_csv(output_loc, index=False)
    print('rows written {}'.format(df.shape[0]))
    return df
