import pickle,yaml, json
from metrics import updateMetricsSheet
import models.models as mdl
import torch,pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from types import SimpleNamespace
weight_types=['pickle','nn']
weight_type=weight_types[1]


with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
if weight_type=='pickle':
    def transform(train):
        if 'target' in list(train.columns):train = train.drop('target', axis=1)
        all_cols = [c for c in list(train.columns) if c not in ['customer_ID', 'S_2']]
        cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
        num_features = [col for col in all_cols if col not in cat_features]
        test_num_agg = train.groupby('customer_ID')[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
        test_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
        train = pd.concat([test_num_agg, test_cat_agg], axis=1)
        train = train.replace([np.inf, -np.inf, np.nan, np.inf], -127)
        return train

    from inference import get_inference
    filename='forest/v2.sav'
    loaded_model = pickle.load(open(config.weight_loc+filename, 'rb'))
    model=loaded_model.predict_proba
    varlist=pd.read_csv(config.data_loc+'/from_radar/playground/6/',nrows=1).columns

    #varlist=None
    transformation=transform
elif weight_type=='nn':
    from transformer_inference import get_inference
    from input_diagnostics.transformations import normalizer,woe_transformation

    with open(config.weight_loc + "norm_dict", "r") as fp:dict = json.load(fp)
    normalize=normalizer(f1=dict)
    #normalizer = lambda x: x
    filename='model_41.pth'#transformer_v1_9.pth'
    model = mdl.__dict__[config.model]
    model = model(input_size=188,output_size=1,dropout=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.weight_loc+filename, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    varlist_loc=['customer_ID', 'target']
    varlist = list(dict['mean'].keys()) + ['customer_ID']
    varlist=list(pd.read_csv(config.data_loc+'/from_radar/playground/6/dev.csv',nrows=1).columns)
    varlist.remove('target')
    #varlist=['customer_ID']+varlist
    transformation=woe_transformation(woe_file=config.output_loc+'eda/feature_importance/')

#apply on train and hold out

if True:
    loc = "from_radar/original_radar/90_10_split/"
    train_pred=get_inference(model,loc=config.data_loc+loc+"hold_out.csv",output_loc=config.output_loc+'test_prediction/train_pred.csv',varlist=varlist,transformation=transformation)
    dev_pred,dev_actual=train_pred['prediction'],train_pred['target']

    hold_pred=get_inference(model,loc=config.data_loc+loc+"hold_out.csv",output_loc=config.output_loc+'test_prediction/hold_out_pred.csv',varlist=varlist,transformation=transformation)
    hold_pred,hold_actual=hold_pred['prediction'],hold_pred['target']
    updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc=config.output_loc+ 'metricSheet.csv', modelName=filename,force=True)
    train_pred,hold_pred=0,0
if True:get_inference(model,loc=config.data_loc+"from_radar/original_radar/"+"test.csv",output_loc=config.output_loc+'test_prediction/submission.csv',varlist=varlist,keep_actual=False,transformation=transformation)
