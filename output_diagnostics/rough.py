import pandas as pd
import numpy as np
from utils.funcs import lorenzCurve
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
if True:
    pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/outputs/test_prediction/submission1.csv')

