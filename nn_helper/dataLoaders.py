import random

from torch.utils.data import Dataset
import pandas as pd
import torch,pickle
import numpy as np
import json
from utils.data_packing import packing
maxrows =5000
class DigitData(Dataset):
    def __init__(self, data_frame=None, label=None, pixel_col=None, reshape_pixel=None, path=None):
        if data_frame is None:
            if maxrows is None:
                data_frame = pd.read_csv(path, nrows=maxrows)
            else:
                data_frame = pd.read_csv(path,nrows=maxrows)
        self.data = data_frame
        self.data.reset_index(inplace=True, drop=True)
        self.labelCol = label
        self.pixelCol = pixel_col
        self.reshape_pixel = reshape_pixel

    def __getitem__(self, idx):
        data = self.data.loc[idx]
        label, pixel = torch.tensor(data[self.labelCol], dtype=torch.long), torch.tensor(data[self.pixelCol]
                                                                                         , dtype=torch.float32)
        pixel = pixel.reshape(self.reshape_pixel)
        pixel = torch.where(pixel > torch.tensor(100), torch.tensor(255.0), torch.tensor(0.0))  # add step to preprocess the data
        pixel = torch.unsqueeze(pixel, dim=0)
        return {'targets': label, 'image_pixels': pixel}

    def __len__(self):
        return self.data.shape[0]

class DigitData_l(DigitData):
    def __init__(self, data_frame=None, label=None, pixel_col=None,localization_col=None, reshape_pixel=None, path=None):
        super().__init__(data_frame,label, pixel_col, reshape_pixel, path)
        self.localization_col = localization_col
    def __getitem__(self, idx):

        out = super().__getitem__(idx)
        out['targets'] =torch.tensor([out['targets'].tolist()]+packing.unpack(self.data.loc[idx][self.localization_col]), dtype=torch.float32)
        return out
class DigitData_mult_object(DigitData):
    def __init__(self, data_frame=None,data_frame_loc=None, label=None, pixel_col=None, localization_col=None, reshape_pixel=None,
                 path=None,path_loc=None):
        super().__init__(data_frame, label, pixel_col, reshape_pixel, path)
        if data_frame_loc is None:
            if maxrows is None:
                data_frame_loc = pd.read_csv(path_loc, nrows=maxrows)
            else:
                data_frame_loc = pd.read_csv(path_loc,nrows=maxrows)
        self.data_loc=data_frame_loc
        self.localization_col = localization_col
    def __getitem__(self, idx):
        """

        :param idx: index for which data to be return
        :return: target as tensor with
        """
        box=[]
        out = super().__getitem__(idx)
        data=self.data_loc[self.data_loc['index']==idx][self.labelCol+self.localization_col]

        for row in data.rows():
            box.append(torch.tensor(row))
        out['targets']=box

        return out
class amex_dataset(Dataset):
    def __init__(self, group=None, n_skill=4 , max_seq=13, dev=True,pickled=True):  # HDKIM 100
        super(amex_dataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill

        self.dev = dev
        #         self.user_ids = [x for x in group.index]
        self.user_ids = []
        if dev :
            if pickled:
                self.group_t = pickle.load(open(group[1], 'rb'))
                group=pickle.load(open(group[0], 'rb'))
            else:
                self.group_t = group[1]
                group = group[0]


        self.samples = group
        for user_id in group.keys():
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):

        user_id = self.user_ids[index]
        tuparray = torch.from_numpy(self.samples[user_id].astype(np.float))
        seq_len = tuparray.shape[0]
        #tuparray = np.array(tup)
        #print(seq_len)

        if seq_len >= self.max_seq:
            outs = tuparray[-self.max_seq:,:]
        else:
            #seq_len=1
            outs = torch.cat((torch.zeros((self.max_seq-seq_len,tuparray.shape[1])),tuparray[-seq_len:,:])) #Todo
        #toDo remove this condition
        # if outs[-1][0]>0:outs[-1,1:12]=torch.rand((11,))-0.5
        # else: outs[-1,-12:]=torch.rand((12,))-0.5
        #print(outs[-1,-12:].mean())

        if self.dev: return {'image_pixels':outs.to(torch.float32),'targets':torch.tensor(self.group_t[index],dtype=torch.float32)}
        else:return {'image_pixels':outs.to(torch.float32),'targets':torch.tensor(0,dtype=torch.float32)}

    @classmethod
    def create_dict(cls,df_loc,key,var_drop=['customer_ID', 'S_2'],target=None,batch=1000,max_row=None,identifier='',save_loc="",var_list=None ):
        #todo check if none row are missed or docuble calcuated
        def breakinUsers(r):
            r = r.drop(var_drop, axis=1)
            return r.to_numpy()
        if isinstance(df_loc,str):
            if max_row is not None:df=pd.read_csv(df_loc,usecols=[key],nrows=max_row)
            else:df=pd.read_csv(df_loc,usecols=[key])
            if target is not None:
                target_dict = pd.read_csv(df_loc, usecols=[key, target], nrows=max_row).groupby('customer_ID').tail(1).set_index(key)[
                    target]

        else :
            df=df_loc
            target_dict=df_loc.groupby('customer_ID').tail(1).set_index(key)[target]
        max_row=df.shape[0]


        rows = df[[key]].drop_duplicates(subset=['customer_ID'], keep='last').index.to_list()
        df=0
        group = pd.Series([])
        loop=1
        from_row = 0
        max_row=rows[-1]+1
        print(max_row)
        if var_list is  None:
            with open(config.weight_loc + "all_var", "r") as fp:var_list = json.load(fp)
        del var_list['S_2']
        var_list=list(var_list.keys())
        while from_row <= max_row:
            if from_row==max_row:break
            elif loop*batch<=len(rows):
                to_row=rows[loop*batch]+1
            else :
                to_row=max_row
            if isinstance(df_loc,str):train = pd.read_csv(df_loc,usecols=var_list+['customer_ID','target'], skiprows=range(1, from_row + 1),
                                nrows=to_row-from_row, header=0)#.groupby(key).tail(1)  #tail1 was done to check attention mechanism is working or not
            else:train=df_loc
            train=train.set_index(key).replace([np.inf,-np.inf,np.nan,np.inf],0.00).select_dtypes(exclude=['object','O']).reset_index() #todo check for better replacement

            if train.shape[0]>1:group=group.append(train.groupby('customer_ID').apply(breakinUsers))
            loop+=1
            print(from_row,to_row)
            from_row = to_row

            if save_loc is not None:
                with open(save_loc +identifier+'dict1.pkl', 'wb') as f:
                    pickle.dump(group.reset_index(drop=True).to_dict(), f)
                if target is not None:
                    with open(save_loc+identifier+'dict2.pkl', 'wb') as f:
                        pickle.dump(target_dict.reset_index(drop=True).to_dict(), f)
        if target is not None:
            return group.reset_index(drop=True).to_dict(),target_dict.reset_index(drop=True).to_dict()
        else: return group.reset_index(drop=True).to_dict()




#     #train_df[train_df['user_id']==115]


if __name__ == "__main__":
    from input_diagnostics.common import *
    import random
    def west1():
        for file in ['dev','hold_out']:
            df_loc='/home/pooja/PycharmProjects/amex_default_kaggle/data/intermediate_data/'+'rad_stan_'+file+'.csv'
            t=amex_dataset.create_dict(df_loc=df_loc,key='customer_ID',var_drop=['customer_ID','target'],target='target',batch=50000,max_row=None,identifier='from_radar/rad_stan_'+file,save_loc=config.data_loc)
        #pickle_jar=[config.data_loc +'data_created/'+'dict1.pkl',config.data_loc +'data_created/'+'dict2.pkl']
        #q=amex_dataset(groupzxc = pickle_jar, n_skill = 4, max_seq = 13, dev = True)
        #d=q.__getitem__(3)
    def west2(): #cheking if pickle files are correct
        df_loc = '/home/pooja/PycharmProjects/amex_default_kaggle/data/data_created/hold_out.csv'

        source=pd.read_csv(df_loc,nrows=100000)
        custs=source['customer_ID'].unique()

        group= [config.data_loc + 'data_created/' + 'v_dict1.pkl',
                      config.data_loc + 'data_created/' + 'v_dict2.pkl']
        group_t = pickle.load(open(group[1], 'rb'))
        group = pickle.load(open(group[0], 'rb'))
        target_cust=random.randint(0,1000)
        source=source[source['customer_ID']==custs[target_cust]]
        dest=group[target_cust]
        dest_val=group_t[target_cust]
        d=0

    west1()




    f=0
    # print(start-stop)