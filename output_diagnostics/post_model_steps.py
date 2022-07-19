import pickle,yaml,pandas as pd,json,numpy as np
from metrics import updateMetricsSheet
import models.models as mdl
import torch


#from sklearn.ensemble import RandomForestClassifier
from types import SimpleNamespace
weight_types=['pickle','nn']
weight_type=weight_types[1]


with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
if weight_type=='pickle':
    from inference import get_pred, get_inference
    filename=""
    loaded_model = pickle.load(open(config.weight_loc+filename, 'rb'))
    model=loaded_model.predict_proba
elif weight_type=='nn':
    from transformer_inference import get_inference
    filename='transformer_v1_3.pth'
    model = mdl.__dict__[config.model]
    model = model(input_size=186, output_size=1, num_blocks=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.weight_loc+filename, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

#apply on train and hold out
varlist_loc=['customer_ID','target']
if True:
    train_pred=get_inference(model,loc=config.data_loc+"data_created/"+"dev.csv",output_loc=config.output_loc+'test_prediction/train_pred.csv',varlist_loc=varlist_loc)
    dev_pred,dev_actual=train_pred['prediction'],train_pred['target']
    hold_pred=get_inference(model,loc=config.data_loc+"data_created/"+"hold_out.csv",output_loc=config.output_loc+'test_prediction/hold_out_pred.csv',varlist_loc=varlist_loc)
    hold_pred,hold_actual=hold_pred['prediction'],hold_pred['target']
    updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc=config.output_loc+ 'metricSheet.csv', modelName="transformer_v1",force=True)
    train_pred,hold_pred=0,0
get_inference(model,loc=config.data_loc+"from_kaggle/"+"test_data.csv",output_loc=config.output_loc+'test_prediction/submission.csv',varlist_loc=['customer_ID'],keep_actual=False)
