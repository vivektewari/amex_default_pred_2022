from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import time,pickle
import numpy as np
import pandas as pd,json
import yaml
from types import SimpleNamespace
from output_diagnostics.metrics import amex_metric
with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))

train=pd.read_csv(config.data_loc+"from_radar/original_radar/"+"dev.csv",nrows=20000)
target=train.groupby('customer_ID').tail(1)['target']

def process(train):
    train = train.drop('target', axis=1)
    all_cols = [c for c in list(train.columns) if c not in ['customer_ID','S_2']]
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    num_features = [col for col in all_cols if col not in cat_features]
    test_num_agg= train.groupby('customer_ID')[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    test_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
    train = pd.concat([test_num_agg, test_cat_agg], axis=1)
    train=train.replace([np.inf,-np.inf,np.nan,np.inf],-127)
    del test_num_agg, test_cat_agg
    print('finish_processing')
    return train
train=process(train)
train2=pd.read_csv(config.data_loc+"from_radar/original_radar/"+"hold_out.csv")
target2=train2.groupby('customer_ID').tail(1)['target']
train2=process(train2)
train=train.append(train2)
target=target.append(target2)
train2,target2=0,0
train3=pd.read_csv(config.data_loc+"from_radar/original_radar/"+"dev.csv",skiprows=range(1,2000000),header=0)
target3=train3.groupby('customer_ID').tail(1)['target']
train3=process(train3)
train=train.append(train3)
target=target.append(target3)
train3,target3=0,0

print('shape after engineering', train.shape )
#train_label=pd.read_csv(config.data_loc+"fr,om_kaggle/"+"train_labels.csv",index_col='customer_ID')
#train=train.join(train_labell[o]6)


consider_len=50
#candidates=pd.read_excel(open(config.output_loc+'eda/feature_importance/'+config.feature_file_name,'rb'),sheet_name='iv_summary')
#iv_set=set(list(candidates.sort_values(['IV'],ascending=False)['variable'])[0:consider_len])
#candidates=pd.read_excel(open(config.output_loc+'eda/feature_importance/'+config.feature_file_name,'rb'),sheet_name='forest_importances')
#forest_set=set(list(candidates.sort_values(['value'],ascending=False)['variable'])[0:consider_len])
#var_list=list(iv_set.intersection(forest_set))
#train=train[var_list]
with open(config.weight_loc +"forest/varlist_v2", "w") as fp:json.dump(list(train.columns), fp)
#train=train.select_dtypes(exclude=['object','O'])

feature_names = [i for i in train.columns]

X_train, y_train=train.to_numpy(),target.to_numpy()
train,target=0,0
start_time = time.time()
if False:
    clasifier = RandomForestClassifier(random_state=0,n_jobs=4)
    clasifier.fit(X_train, y_train)
print('starting _training')
if True:
    xgb_params = {
        'max_depth': 8,
        'learning_rate': 0.15,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'early_stopping_rounds':150}
    clasifier = xgb.XGBClassifier(num_boost_round=9999,**xgb_params)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kfold.split(X_train, y_train):
        print(len(train_index), len(test_index))
        x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        X_train, y_train=0,0
        clasifier.fit(x_train_fold,y_train_fold, eval_set=[(x_test_fold,y_test_fold )])
        pred=clasifier.predict(x_test_fold)
        #print("amex_metrics {}".format(amex_metric(y_test_fold,pred)))
        break

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
print('done _training')
filename =config.weight_loc+'forest/v2.sav'
pickle.dump(clasifier, open(filename, 'wb'))
importances = clasifier.feature_importances_
#std = np.std([tree.feature_importances_ for tree in clasifier.estimators_], axis=0)


forest_importances = pd.Series(importances, index=feature_names)
forest_importances.to_csv(config.weight_loc +"forest/imp.csv")

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

