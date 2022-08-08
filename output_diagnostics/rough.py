import pandas as pd
import numpy as np
from utils.funcs import lorenzCurve
from input_diagnostics.common import *
if False:
    sub1=pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/test_prediction/submission1.csv')
    sub2=pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/test_prediction/submission.csv')
    sub=sub1.append(sub2)
    sub=sub.drop_duplicates(['customer_ID'],keep='last')
    submission=pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/data/from_kaggle/sample_submission.csv',usecols=['customer_ID'])
    submission=submission.join(sub.set_index('customer_ID'),on=['customer_ID'])
    print(submission['prediction'].isnull().sum(),submission.shape[0])
    submission['prediction']=submission['prediction'].replace([np.nan],value=0.0)

    submission[['customer_ID','prediction']].to_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/test_prediction/kaggle_submission.csv',index=False)
if False:
    pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/test_prediction/submission1.csv')
if False:
    loc='test_prediction/7/'
    from utils.funcs import lorenzCurve
    df=pd.read_csv(config.output_loc+loc+'hold_out_pred.csv')
    zeros=df[df['target']==0]
    for i in range(19):# appending to make ratio 1:20 for 1:0
        df=df.append(zeros)
    lorenzCurve(df['target'],df['prediction'],save_loc=config.output_loc+loc+'lorenz')

if 1:
    with open(config.data_loc + "data_created/rough/inference_tensor", "r") as fp: var_list = json.load(fp)
    with open(config.data_loc + "data_created/rough/trainer_tensor", "r") as fp: var_list2 = json.load(fp)
    for i in range(13):
        for j in range(188):
            if abs(var_list2[i][j]-var_list[i][j])>0.001:
                v=0