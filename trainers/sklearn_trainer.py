from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import time,pickle
import numpy as np
import pandas as pd,json
import yaml
from types import SimpleNamespace
with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))

train=pd.read_csv(config.data_loc+"data_created/"+"dev.csv")#,nrows=10000,
train= train.groupby('customer_ID').tail(1).set_index('customer_ID')
#train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')
#train=train.join(train_label)
target=train['target']
consider_len=50
candidates=pd.read_excel(open(config.output_loc+'eda/feature_importance/'+config.feature_file_name,'rb'),sheet_name='iv_summary')
iv_set=set(list(candidates.sort_values(['IV'],ascending=False)['variable'])[0:consider_len])
candidates=pd.read_excel(open(config.output_loc+'eda/feature_importance/'+config.feature_file_name,'rb'),sheet_name='forest_importances')
forest_set=set(list(candidates.sort_values(['value'],ascending=False)['variable'])[0:consider_len])
var_list=list(iv_set.intersection(forest_set))
train=train[var_list]
with open(config.weight_loc +"forest/varlist_v2", "w") as fp:json.dump(var_list, fp)
train=train.select_dtypes(exclude=['object','O'])
train=train.replace([np.inf,-np.inf,np.nan,np.inf],0.00)
feature_names = [i for i in train.columns]
X_train, y_train=train.to_numpy(),target.to_numpy()

forest = RandomForestClassifier(random_state=0,n_jobs=4)
print('starting _training')
start_time = time.time()
forest.fit(X_train, y_train)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
print('done _training')
filename =config.weight_loc+'forest/v2.sav'
pickle.dump(forest, open(filename, 'wb'))
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)


forest_importances = pd.Series(importances, index=feature_names)


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

