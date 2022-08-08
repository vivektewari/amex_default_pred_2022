from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
import yaml,os
from types import SimpleNamespace
from common import *

start_time = time.time()
train_feature=pd.read_csv(config.data_loc+"from_radar/"+"dev.csv")
train_feature= train_feature.groupby('customer_ID').tail(1).set_index('customer_ID')

#train_label=pd.read_csv(config.data_loc+"from_radar/"+"train_labels.csv",index_col='customer_ID')
train_label=train_feature['target']
#=train_feature.drop('target',axis=1)


df=train_feature.select_dtypes(exclude=['object','O'])
df=df.replace([np.inf,-np.inf,np.nan],0.00)#.notnull().all(axis=1)]
#df=df.join(train_label)
X_train, y_train=df.to_numpy(),df['target'].to_numpy()
feature_names = [i for i in df.columns]
forest = RandomForestClassifier(random_state=0,n_jobs=4)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.DataFrame({'variable':feature_names,'value':importances})
writer = pd.ExcelWriter(config.output_loc+'eda/feature_importance/'+config.feature_file_name,mode='a')
forest_importances.to_excel(writer,sheet_name="forest_importances",index=False)
writer.save()
writer.close()

#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()