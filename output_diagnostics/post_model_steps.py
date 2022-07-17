import pickle,yaml,pandas as pd,json,numpy as np
from metrics import updateMetricsSheet
from inference import  get_pred,get_inference
#from sklearn.ensemble import RandomForestClassifier
from types import SimpleNamespace
weight_types=['pickle']
weight_type=weight_types[0]
filename='forest/v2.sav'
with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
if weight_type=='pickle':
    loaded_model = pickle.load(open(config.weight_loc+filename, 'rb'))
    model=loaded_model.predict_proba
#apply on train and hold out

if True:
    train_pred=get_inference(model,loc=config.data_loc+"data_created/"+"dev.csv",output_loc=config.output_loc+'test_prediction/train_pred.csv',varlist_loc=config.weight_loc +"forest/varlist_v2")
    dev_pred,dev_actual=train_pred['prediction'],train_pred['target']
    hold_pred=get_inference(model,loc=config.data_loc+"data_created/"+"hold_out.csv",output_loc=config.output_loc+'test_prediction/hold_out_pred.csv',varlist_loc=config.weight_loc +"forest/varlist_v2")
    hold_pred,hold_actual=hold_pred['prediction'],hold_pred['target']
    updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc=config.output_loc+ 'metricSheet.csv', modelName="forest_v2", extraInfo="var_selection_50",force=True)
    train_pred,hold_pred=0,0
get_inference(model,loc=config.data_loc+"from_kaggle/"+"test_data.csv",output_loc=config.output_loc+'test_prediction/submission.csv',varlist_loc=config.weight_loc +"forest/varlist_v2",keep_actual=False)
