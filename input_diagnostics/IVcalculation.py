import warnings
from dataExploration import distReports,plotGrabh
from iv import IV
from common import *
import pandas as pd

cols=['customer_ID', 'S_3', 'P_2', 'D_48', 'D_61', 'D_55', 'D_62', 'B_17', 'D_77', 'D_53', 'S_7', 'P_3', 'D_43', 'R_27', 'D_42', 'D_46', 'D_56', 'S_27', 'D_50', 'D_132', 'D_130', 'D_131', 'D_115', 'D_76', 'B_13', 'D_118', 'D_119', 'D_121', 'D_128', 'D_134', 'D_142', 'D_105', 'target']
with open(config.weight_loc + "all_var2", "r") as fp:cols = json.load(fp)
cols=cols+['customer_ID']+['target']
if 0:
    train=pd.DataFrame()
    for i in range(5):

        if i == 0:
            temp=pd.read_csv(config.data_loc+"from_radar/original_radar/90_10_split/"+"dev.csv",usecols=cols,nrows=1000000,header=0).groupby('customer_ID').tail(4).reset_index().drop('customer_ID', axis=1)
        else:
            temp = pd.read_csv(config.data_loc + "from_radar/original_radar/90_10_split/" + "dev.csv", usecols=cols,
                           nrows=1000000, header=0, skiprows=(1, 1000000 * i)).groupby('customer_ID').tail(
            4).reset_index().drop('customer_ID', axis=1)
        train= train.append(temp)

        temp=0
        print(train.shape)
    train=train.reset_index().drop(['index','level_0'],axis=1)

#only doing on ast observatiion
if 0:

    a = IV(getWoe=1, verbose=1,sort_feature=None)
    #train = train.groupby('customer_ID').tail(4).reset_index().drop('customer_ID', axis=1)
    train=a.binning(train,'target',maxobjectFeatures=50,varCatConvert=1,qCut=40)
    ivData=a.iv_all(train,'target')
    a.saveVarcards(config.output_loc+'eda/feature_importance/')
#binned.to_csv(config.data_loc+'intermediate_data/binned.csv')
if 0:
    for file in ['dev.csv','hold_out.csv']:
        train = pd.read_csv(config.data_loc + "from_radar/original_radar/90_10_split/" + file, usecols=cols)  #
        a=IV(verbose=0)
        a.load(config.output_loc+'eda/feature_importance/')
        a.excludeList=['customer_ID','target']
        converted=a.convertToWoe(train)
        converted['customer_ID']=train['customer_ID']
        converted['target'] = train['target']
        converted.to_csv(config.data_loc+'from_radar/playground/7/'+file,index=False)
        converted,train=0,0#free up memmory
if 1:#woe implementation
    from transformations import woe_transformation

    cols = list(pd.read_csv(config.data_loc + '/from_radar/playground/6/dev.csv', nrows=1).columns)
    #varlist.remove('target')
    transformation=woe_transformation(woe_file=config.output_loc+'eda/feature_importance/',varlist=cols)
    for file in ['hold_out.csv','dev.csv', ]:
        transformation.output_loc=config.data_loc+'from_radar/playground/7/'+file
        transformation(config.data_loc + "from_radar/original_radar/90_10_split/" + file)


if  0:
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
