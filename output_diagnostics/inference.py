import pandas as pd,numpy as np
import random,json


random.seed(24)
def get_pred(model,train,varlist,multi_class=True,transformation=None):
    #train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')
    if 'target' in train.columns:target=train.groupby('customer_ID').tail(1)['target'].to_numpy()
    else :target=np.zeros(train.groupby('customer_ID').tail(1).shape[0])
    with open(varlist, "r") as fp:var_list =json.load(fp)
    if transformation is not None:train=transformation(train)
    train=train.replace([np.inf,-np.inf,np.nan,np.inf],0.00)


    train=train[var_list]
    X_train= train.to_numpy()
    pred=model(X_train)
    if multi_class:pred=pred[:,1]
    return pred,target
def get_inference(model,loc="",output_loc="",varlist="",keep_actual=True,pred_rows=500000,transformation=None):
    max_row=pd.read_csv(loc,usecols=['customer_ID']).shape[0]
    print('rows read {}'.format(max_row))
    from_row=0
    to_row=min(from_row+pred_rows,max_row)
    pd.DataFrame(columns=['customer_ID','prediction','target']).to_csv(output_loc,index=False)

    while from_row<=max_row:
        test_file = pd.read_csv(loc,skiprows=range(1,from_row+1),nrows=pred_rows,header=0)

        pred,actual=get_pred(model,test_file[:],varlist,transformation=transformation)
        test_file= test_file.groupby('customer_ID').tail(1)#.set_index('customer_ID')
        test_file['prediction'] = pred
        test_file['target'] = actual
        test_file[['customer_ID', 'prediction','target']].to_csv(output_loc,index=False,mode='a',header=False)
        from_row=to_row
        print('prediction completed for {}'.format(to_row))
        to_row=pred_rows+to_row
    df = pd.read_csv(output_loc)
    if not keep_actual:
        df=df[['customer_ID', 'prediction']]
    df = df.drop_duplicates(['customer_ID'], keep='last')
    df.to_csv(output_loc, index=False)
    print('rows written {}'.format(df.shape[0]))
    return df.reset_index()

