from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

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
    def __init__(self, group, n_skill, max_seq=MAX_SEQ, dev=True):  # HDKIM 100
        super(amex_dataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        self.dev = dev
        #         self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        # print(time.time()-start,2.0)
        feature = 2 + featureFinal  # len(tup[])##################chnage
        user_id = self.user_ids[index]
        # print(user_id)
        tup = self.samples[user_id]
        seq_len = len(tup[0])
        tuparray = np.array(tup)
        outs = np.zeros((feature + 2, self.max_seq), dtype='float32')
        outs[3, :] = -1
        if seq_len >= self.max_seq:
            outs[2:, :] = tuparray[:, -self.max_seq:]
        else:
            outs[2:, -seq_len:] = tuparray

        # print(time.time()-start,2.4)
        target_id = outs[2, 1:].copy()
        x = outs[2, :-1].copy()
        x += (outs[3, :-1] == 1) * self.n_skill
        if self.dev:
            label = outs[3, 1:]
            label, weights = np.where(label == -1, 0, label), np.where(label == -1, 0, 1)
            outs = outs[:, 1:]
            outs[2, :] = label
            outs[3, :] = weights

        else:
            outs = outs[:, 1:]
            outs[2, :] = 0
            outs[3, :] = 0

        outs[0, :] = x
        outs[1, :] = target_id

        # ret=[x, target_id, label,weights]+list(outs[2:,:])

        #         #print((time.time()-start)*1000,2.6)
        #         #print(time.time()-start,2.6)
        return tuple(outs)  # x, target_id, label,weights,t,lect_count,relevantLectattended,quest_roll_average_50


#     #train_df[train_df['user_id']==115]


if __name__ == "__main__":
