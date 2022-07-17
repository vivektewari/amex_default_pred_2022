import warnings
from dataExploration import distReports,plotGrabh
from iv import IV
from common import *
import pandas as pd

a=IV(getWoe=0,verbose=True)
train_feature=pd.read_csv(config.data_loc+"data_created/"+"dev.csv",nrows=100000)
train_label=pd.read_csv(config.data_loc+"from_kaggle/"+"train_labels.csv",index_col='customer_ID')
train= train_feature.groupby('customer_ID').tail(1).set_index('customer_ID')
#train=train_feature.join(train_label)
#only doing on ast observatiion
train=train.reset_index().drop('customer_ID',axis=1)
binned=a.binning(train,'target',maxobjectFeatures=300,varCatConvert=1)
#binned.to_csv(config.data_loc+'intermediate_data/binned.csv')

ivData=a.iv_all(binned,'target')

#ivData =pd.read_csv(config.output_loc+'eda/feature_importance/'+"iv.csv")
writer = pd.ExcelWriter(config.output_loc+'eda/feature_importance/'+config.feature_file_name)
ivData.to_excel(writer,sheet_name="iv_detailed")
ivData.groupby('variable')['IV'].sum().to_excel(writer,sheet_name="iv_summary")
#ivData.to_csv(loc+"iv_detailed_cross.csv")
#ivData.groupby('variable')['ivValue'].sum().to_csv(loc+"iv_sum_cross2.csv")

# ivInfo=pd.read_csv(loc+"iv3.csv")
# distRepo=distReports(train,ivInfo)
# distRepo.to_csv(loc+"summary.csv")
writer.save()
writer.close()
