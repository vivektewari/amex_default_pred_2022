import pandas as pd
import json
import yaml
from output_diagnostics.metrics import updateMetricsSheet
from types import SimpleNamespace
with open('./config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))

import random
task=['sample_index_creation','holdOutSample' ,'randomModel']
stage=task[1]

if stage=='sample_index_creation':
    #train = pd.read_csv(config.data_loc+ '/from_kaggle/train_data.csv')
    train = pd.read_csv(config.data_loc + '/from_kaggle/train_labels.csv')
    # train['index']=range(train.shape[0])
    # orderedCols=list(train.columns[-1:])+list(train.columns[0:-1])
    # train=train[orderedCols]
    breakParts=5
    rows=random.sample(range(train.shape[0]), int(train.shape[0]/breakParts))
    rows=list(set(rows)-set([i for i in range(100)]))# keepint this elemnet in training to visulize these lement
    remaining=list(set(list(range(train.shape[0])))-set(rows))
    #duping list to json
    with open(config.data_loc +"data_created/dev_index", "w") as fp:json.dump(rows, fp)
    with open(config.data_loc + "data_created/hold_out_index", "w") as fp:json.dump(remaining, fp)

elif stage=='holdOutSample':
    train = pd.read_csv(config.data_loc + '/from_kaggle/train_data.csv',nrows=3000000)
    train_labels = pd.read_csv(config.data_loc + '/from_kaggle/train_labels.csv',index_col='customer_ID')
    train=train.join(train_labels,on='customer_ID')

    with open(config.data_loc + "data_created/hold_out_index", "r") as fp:remaining =json.load(fp)

    remaining=[r for r in remaining if r <3000000 ]
    train.iloc[remaining].to_csv(config.data_loc + 'data_created/hold_out.csv', index=False)
    train=train.drop(remaining,axis=0)
    train.to_csv(config.data_loc + 'data_created/dev.csv', index=False)

elif stage=='randomModel':
    dev=pd.read_csv(config.data_loc + 'data_created/dev.csv',nrows=1000)
    holdOut=pd.read_csv(config.data_loc + 'data_created/hold_out.csv',nrows=1000)
    predDev=[random.random() for i in range(dev.shape[0])]
    predHold = [random.random() for i in range(holdOut.shape[0])]
    updateMetricsSheet(dev['target'],predDev,holdOut['target'],predHold,modelName='random',force=False,loc=config.output_loc+'metricSheet.csv')





