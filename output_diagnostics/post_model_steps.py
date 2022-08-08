import pickle,yaml, json
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
    from inference import get_inference
    filename='forest/v2.sav'
    loaded_model = pickle.load(open(config.weight_loc+filename, 'rb'))
    model=loaded_model.predict_proba
    varlist_loc=config.weight_loc +"forest/varlist_v2"
elif weight_type=='nn':
    from transformer_inference import get_inference
    from input_diagnostics.transformations import normalizer

    with open(config.weight_loc + "norm_dict", "r") as fp:dict = json.load(fp)
    normalize=normalizer(f1=dict)
    #normalizer = lambda x: x
    filename='transformer_v1_12.pth'
    model = mdl.__dict__[config.model]
    model = model(input_size=188,output_size=1,dropout=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.weight_loc+filename, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    varlist_loc=['customer_ID', 'target']

#apply on train and hold out

if True:
    loc = "from_radar/"
    train_pred=get_inference(model,loc=config.data_loc+loc+"dev.csv",output_loc=config.output_loc+'test_prediction/train_pred.csv',varlist=list(dict['mean'].keys())+['customer_ID'],transformation=normalize)
    dev_pred,dev_actual=train_pred['prediction'],train_pred['target']


    hold_pred=get_inference(model,loc=config.data_loc+loc+"hold_out.csv",output_loc=config.output_loc+'test_prediction/hold_out_pred.csv',varlist=list(dict['mean'].keys())+['customer_ID'],transformation=normalize)
    hold_pred,hold_actual=hold_pred['prediction'],hold_pred['target']
    updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc=config.output_loc+ 'metricSheet.csv', modelName=filename,force=True)
    train_pred,hold_pred=0,0
if True:get_inference(model,loc=config.data_loc+"from_radar/"+"test.csv",output_loc=config.output_loc+'test_prediction/submission.csv',varlist=list(dict['mean'].keys())+['customer_ID'],keep_actual=False,transformation=normalize)
