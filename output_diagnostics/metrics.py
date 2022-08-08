import pandas as pd,json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import numpy as np
from datetime import datetime
import torch
from multiprocessing import Pool, cpu_count


pool=Pool(4)
def bad_capture(y,x,perc=4):
    x_array=np.array(x)
    boundary_val=np.percentile(x_array,100-perc)
    return y[np.argwhere(x_array >boundary_val).flatten()].sum()/y.sum()


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    y_true=pd.DataFrame({'target':y_true})
    y_pred= pd.DataFrame({'prediction': y_pred})
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d),g,d
def getMetrics(actual, predicted):
    """
    :param actual: actual series
    :param predicted: predicted series
    :return: list of accuracy ,precision ,recall and f1
    """
    met = []
    metrics = [roc_auc_score,bad_capture,amex_metric]#accuracy_score, precision_score, recall_score, f1_score]
    for m in metrics:
        if m == accuracy_score:
            met.append(m(actual, predicted))
        elif m==amex_metric:
            temp=m(actual, predicted)
            for i in range(3):met.append(temp[i])
        else:
            met.append(m(actual, predicted))
    return met



def updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc="", modelName="", extraInfo="",
                       force=False):
    model = 'model'
    f = pd.read_csv(loc,index_col='index')
    if modelName in list(f[model]):
        if not force: raise Exception("model exist. try with Force as True or different model name")
        # else:f.drop(f[f[model]==modelName].index,axis=0)
    metricsDev, metricsHold = getMetrics(dev_actual, dev_pred), getMetrics(hold_actual, hold_pred)
    entryVal = datetime.now(),modelName, *metricsDev, *metricsHold, extraInfo
    dict = {}


    for i in range(f.shape[1]-1):
        dict[f.columns[i]] = entryVal[i]

    pd.DataFrame(dict, index=[f.shape[0]]).to_csv(loc, mode='a', header=False)

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