import pandas as pd
from dataExploration import *
import json
from common import *
if False:
    #anylyzing test and train dataset so can develop dev and holdout sample in same way.
    # This will ensure consistency of clculated metrics of test and hold out sample
    # Q1. is customer_id in test and train are mutually exclusive
    # Q2. for any particular customer_id does all time records (13 or 18 month are present in both
    #A2:train and test are same , for each customer it is a monthly data but day is not end of month day


    # train =pd.read_csv(config.data_loc+'from_kaggle/train_data.csv',nrows=1000)
    # test =pd.read_csv(config.data_loc+'from_kaggle/test_data.csv',nrows=1000)
    train_c =pd.read_csv(config.data_loc+'from_kaggle/train_data.csv',usecols=['customer_ID'])
    test_c =pd.read_csv(config.data_loc+'from_kaggle/train_data.csv',usecols=['customer_ID'])
    intersection=set(train_c['customer_ID']).intersection(set(test_c['customer_ID']))
    print(len(intersection))
    c=0
if 0:


    df=pd.read_csv(config.data_loc+'data_created/dev.csv',nrows=1000000)
    distReports(df,detail=True).to_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/eda/features.csv')
if 0:#radar dataset
    df=pd.read_parquet(config.data_loc+'from_radar/train.parquet')
    distReports(df, detail=True).to_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/eda/radar_features.csv')

    c=0
if 0: # breaking radar dataset in to dev and holdout with same customer_id as ealier
    df=pd.read_parquet(config.data_loc+'from_radar/train.parquet')
    for d in ['dev.csv','hold_out.csv']:
        df_temp=pd.read_csv(config.data_loc+'data_created/'+d,usecols=['customer_ID','target']).groupby('customer_ID').tail(1)
        temp=df[df['customer_ID'].isin(list(df_temp['customer_ID'].unique()))]
        temp=temp.join(df_temp.set_index('customer_ID'),on='customer_ID')
        temp.to_csv(config.data_loc+'from_radar/'+d)
if 0:# making test dataset from parquet
    df = pd.read_parquet(config.data_loc + 'from_radar/test.parquet').to_csv(config.data_loc+'from_radar/'+'test.csv')
if 0:#visulizing radar detaaset
    pass

    iv_report=pd.read_excel(config.output_loc+'eda/feature_importance/feature_importance_v3.xlsx',index_col='variable',sheet_name='iv_summary')
    df = pd.read_csv(config.data_loc + 'from_radar/dev.csv')
    distReports(df,ivReport=iv_report, detail=True).to_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/eda/radar_features.csv')
if 0: # getting float variable with no missing and IV >0.1
    rf=pd.read_csv(config.output_loc+'eda/radar_features.csv')
    filtered=list(rf[rf['dtypes']=='float64'][rf['missing_percent']==0][rf['IV']>0.05][rf['nuniques']>300].varName)
    with open(config.weight_loc +"varlist_pilot", "w") as fp:json.dump(filtered, fp)
if 0:# standerizing above list
    rf = pd.read_csv(config.output_loc + 'eda/radar_features.csv',index_col='varName')
    dict=rf[['mean','std']].to_dict()
    del dict['S_2']

    with open(config.weight_loc +"all_var", "r") as fp: var_list = json.load(fp)
    var_list=list(var_list.keys())
    var_list.remove('S_2') # as this is date variable
    for file in ['dev.csv','hold_out.csv']:
        df=pd.read_csv(config.data_loc +'from_radar/'+file,usecols=var_list+['customer_ID','target'],nrows=2000000)
        for v in var_list:
            df[v]=(df[v]-dict['mean'][v])/dict['std'][v]
        df.to_csv(config.data_loc+'intermediate_data/rad_stan_' +file)
        distReports(df, detail=True).to_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/eda/radar_features_trun_stan.csv')
if 1:# combine all metric sheets t get the best tranformation for each variable
    import os
    loc=config.rough_loc+'metrices/'
    files = os.listdir(loc)
    count=0
    for file in files:
        if ".csv" in file:
            if count==0:df=pd.read_csv(loc+file)
            else :df=df.append(pd.read_csv(loc+file))
            count+=1
    df=df.drop_duplicates(['variable','transformation'])
    df.to_csv(config.rough_loc+'combined_metric_file.csv')
if 1: # from above computing winner tranformation of each varaible
    loc=config.rough_loc+'combined_metric_file.csv'
    df =pd.read_csv(loc)
    #var_list=list(df[df['transformation']=='Slicer Woe|data_consideration:13|']['variable'])
    #df=df[df['variable'].isin(var_list)]
    comparison_metric='auc' #valid_loss'
    df['index']=1
    df=df.sort_values(['variable',comparison_metric],ascending=False)
    df['rank']=df.groupby(['variable'])['index'].cumsum()#.reset_index()['index']
    df.drop_duplicates(['variable']).to_csv(config.rough_loc+comparison_metric+'_transformation_winners.csv')
    df3=df3.groupby('transformation')['rank'].count()
    # get winner for each transformation
    df4=df.groupby('transformation')['rank'].count() # fought for
    df5=df3/df4
    df5=df5.fillna(0)
    df5.to_csv(config.rough_loc+comparison_metric+'_transformation_analysis.csv')
if 0: #getting dictionary correction
    rf = pd.read_csv(config.output_loc + 'eda/radar_features.csv', index_col='varName')
    dict = rf[['mean', 'std']].to_dict()


    with open(config.weight_loc+"all_var2", "r") as fp: var_list = json.load(fp)
    remove_list=[]
    for key in dict['mean'].keys():
        if key not in var_list:
            remove_list.append(key)
    for key in remove_list:
        del dict['mean'][key]
        del dict['std'][key]
    print(len(dict['mean'].keys()))
    with open(config.weight_loc + "norm_dict", "w") as fp: json.dump(dict, fp)


if 1 :
    pass




