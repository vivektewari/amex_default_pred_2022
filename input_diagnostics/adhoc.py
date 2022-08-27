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

    df=pd.read_csv(config.data_loc+'from_radar/playground/6/'+'/dev.csv')
    distReports(df,detail=True).to_csv(config.data_loc+'from_radar/playground/6/'+'dist.csv')
if 0:#radar dataset
    df=pd.read_csv(config.data_loc+'intermediate_data/transformed_data/dev.csv')
    distReports(df, detail=True).to_csv(config.data_loc+'intermediate_data/transformed_data/dev_dist.csv')

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
    #rf = pd.read_csv(config.output_loc + 'eda/radar_features.csv',index_col='varName')
    rf = pd.read_csv(config.data_loc+'from_radar/'+'radar_dist.csv', index_col='varName')
    dict=rf[['mean','std']].to_dict()
    del dict['mean']['S_2']
    del dict['std']['S_2']

    with open(config.weight_loc +"all_var", "r") as fp: var_list = json.load(fp)
    var_list=list(var_list.keys())
    var_list.remove('S_2') # as this is date variable
    for file in ['dev.csv','hold_out.csv']:
        df=pd.read_csv(config.data_loc+'from_radar/original_radar/'+file,usecols=var_list+['customer_ID','target'],nrows=2000000)
        for v in var_list:
            df[v]=(df[v]-dict['mean'][v])/dict['std'][v]
        df.to_csv(config.data_loc+'from_radar/' +file)
        distReports(df, detail=True).to_csv(config.data_loc+'from_radar/dist_'+file)
if 0:# combine all metric sheets t get the best tranformation for each variable
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
if 0: # from above computing winner tranformation of each varaible
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


if 0 :#replacing some columns th woe ones
        var_sel=['B_6','D_62','P_3','R_27']
        with open(config.weight_loc + "all_var2", "r") as fp:
            var_list = json.load(fp)

        for file in ['dev.csv','hold_out.csv']:

            dest = pd.read_csv(config.data_loc + 'from_radar/'  + file,
                             usecols=var_list + ['customer_ID', 'target'])

            source=pd.read_csv(config.data_loc + 'intermediate_data/woe_stan_'+ file,
                             usecols=var_sel + ['customer_ID', 'target'])
            #source=source.sort_values(['customer_ID'])
            for v in var_sel:
                dest[v]=source[v]
            dest.to_csv(config.data_loc + 'intermediate_data/incremental_'+ file,index=False)

if 0:
    with open(config.weight_loc + "all_var", "r") as fp: var_list = json.load(fp)
    var_list = list(var_list.keys())
    var_list.remove('S_2')
    df = pd.read_csv(config.data_loc + 'from_radar/original_radar/dev.csv',nrows=100)
    cols=df.columns
    loop=0
    base_dict={}
    for v in cols:
        if v in var_list:
            if v[0] not in base_dict.keys():
                base_dict[v[0]]=set([loop])
            else :
                base_dict[v[0]].add(loop)
            loop+=1
    d=0

if 0: # missing logic imputation
    #from transformations import missing_imputation

    with open(config.weight_loc + "all_var2", "r") as fp: var_list = json.load(fp)
    missing_rep_dic = pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/planning/missing_imput.csv',index_col=['var'])\
        .to_dict()['decisioning']
    rf = pd.read_csv(config.output_loc + 'eda/radar_features.csv', index_col='varName')
    dict = rf[['mean', 'std']].to_dict()

    mi=missing_imputation(missing_rep_dict=missing_rep_dic,mean_std_dict=dict,output_loc=config.data_loc+ 'from_radar/cleaned/dev.csv',standardize=1,all_var=var_list)
    pd.DataFrame(columns=['customer_ID']  + var_list+['target']).to_csv(mi.output_loc, index=False)
    mi(config.data_loc + 'from_radar/original_radar/dev.csv')
    mi.output_loc=config.data_loc+ 'from_radar/cleaned/hold_out.csv'
    pd.DataFrame(columns=['customer_ID']  + var_list+['target']).to_csv(mi.output_loc, index=False)
    mi(config.data_loc + 'from_radar/original_radar/hold_out.csv')
if 0:# validating difference in trainer sample vs pot model step sample reult
    from output_diagnostics.metrics import amex_metric
    file=pd.read_csv(config.output_loc+'test_prediction/hold_out_pred1.csv',nrows=165919)[-2079*0:]
    print(amex_metric(file['target'],file['prediction']))
if 0:# 90/10 split of orginal data
    import random

    df = pd.read_csv(config.data_loc + 'from_radar/original_radar/dev.csv').drop('Unnamed: 0', axis=1)
    df.to_csv(config.data_loc+'from_radar/original_radar/90_10_split/dev.csv',index=False)
    df=pd.read_csv(config.data_loc+'from_radar/original_radar/hold_out.csv').drop('Unnamed: 0',axis=1)

    cust_list=list(df['customer_ID'].unique())
    random.shuffle(cust_list)
    count = len(cust_list)
    dev=cust_list[:int(count*0.8)]
    hold_out = cust_list[int(count*0.8):]
    df[df['customer_ID'].isin(hold_out)].to_csv(config.data_loc+'from_radar/original_radar/90_10_split/hold_out.csv',index=False)
    df[df['customer_ID'].isin(dev)].to_csv(config.data_loc + 'from_radar/original_radar/90_10_split/dev.csv',index=False,header=False,mode='a')

if 1: #distreports of mew
    for i in range(4):
        df = pd.read_csv(config.data_loc + 'from_radar/playground/7/' + '/dev.csv',skiprows=range(1,2000000+i*500000),header=0,nrows=500000)
        distReports(df, detail=True).to_csv(config.data_loc + 'from_radar/playground/7/'+str(i) + 'dist.csv')





