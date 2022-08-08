import warnings
from dataExploration import distReports,plotGrabh
from iv import IV
from common import *
import pandas as pd


train=pd.read_csv(config.data_loc+"from_radar/"+"dev.csv",nrows=1500000)#

#only doing on ast observatiion
if 1:

    a = IV(getWoe=1, verbose=1,sort_feature=None)
    train = train.groupby('customer_ID').tail(1).reset_index().drop('customer_ID', axis=1)
    binned=a.binning(train,'target',maxobjectFeatures=100,varCatConvert=1,qCut=10)
    ivData=a.iv_all(binned,'target')
    a.saveVarcards(config.output_loc+'eda/feature_importance/')
#binned.to_csv(config.data_loc+'intermediate_data/binned.csv')
if 0:
    a=IV(verbose=1)
    a.load(config.output_loc+'eda/feature_importance/')
    a.excludeList=['customer_ID','target']
    converted=a.convertToWoe(train)
    converted['customer_ID']=train['customer_ID']
    converted['target'] = train['target']
    converted.to_csv(config.data_loc+'from_radar/v_woed.csv',index=False)



if 1:
    #ivData =pd.read_csv(config.output_loc+'eda/feature_importance/'+"iv.csv")
    writer = pd.ExcelWriter(config.output_loc+'eda/feature_importance/'+config.feature_file_name)
    ivData.to_excel(writer,sheet_name="iv_detailed")
    #ivData['Bad_Rate']=ivData.apply(lambda x: if x[''])
    iv_data_sum=ivData.groupby('variable')['IV','Rank_Order'].agg(IV=('IV','sum'),bin_count=('IV','count'),Rank_Order_mean=('Rank_Order','mean'))
    iv_data_sum['Rank_Order']=iv_data_sum.apply(lambda x: x['Rank_Order_mean']>=0.8 or x['Rank_Order_mean']<=0.2,axis=1)
    iv_data_sum.to_excel(writer,sheet_name="iv_summary")

    #ivData.to_csv(loc+"iv_detailed_cross.csv")
    #ivData.groupby('variable')['ivValue'].sum().to_csv(loc+"iv_sum_cross2.csv")

    # ivInfo=pd.read_csv(loc+"iv3.csv")
    # distRepo=distReports(train,ivInfo)
    # distRepo.to_csv(loc+"summary.csv")
    writer.save()
    writer.close()
