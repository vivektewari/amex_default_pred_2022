import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from input_diagnostics.common import *
import cv2
import matplotlib.pyplot as plt
torch.manual_seed(123)
from models import *
class FFN(nn.Module):
    def __init__(self, input_size=200, final_size=200,activation=True):
        super(FFN, self).__init__()
        self.state_size = input_size
        #self.layer_normal = nn.LayerNorm(input_size)
        self.lr1 = nn.Linear(input_size, final_size)
        self.activation= nn.ReLU()#nn.LeakyReLU()
        self.dropout = nn.Dropout(0)
        self.apply_activation=activation

    def forward(self, x):
        x = self.lr1(x)
        if self.apply_activation:x = self.activation(x)
        return self.dropout(x)
class transformer_encoder_block_v2(nn.Module):
    def __init__(self,input_size,output_size,num_heads=2,dropout=0.2,number=0):
        super(transformer_encoder_block_v2, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal0 = nn.LayerNorm(input_size)
        self.layer_normal1 = nn.LayerNorm(input_size)
        self.ffn1 = FFN(input_size,input_size)
        self.ffn2 = FFN(input_size, input_size,activation=False)
        self.number=number

    def forward(self, x):

        x=self.layer_normal0(x)
        att_output, att_weight = self.multi_att(key=x, query=x,value= x)


        for i in range(att_weight.shape[0]):
            att_weight_ = att_weight[i].detach().numpy()
            # att_weight_ = (att_weight_-np.mean(att_weight_) )/ np.std(att_weight_)
            vmax, vmin = np.max(att_weight_), np.min(att_weight_)
            fig = plt.figure()
            plt.imshow(att_weight_, vmin=vmin, vmax=vmax)
            fig.savefig('/home/pooja/PycharmProjects/amex_default_kaggle/data/debugging/att_weight' + str(i) + str(
                self.number) + '__' + str(vmin) + '_' + str(vmax) + '.png', bbox_inches='tight')
            plt.close(fig)
        x = self.layer_normal1(att_output + x)
        x = self.ffn1(self.ffn2(x)) + x

        return x, att_weight


class transformer_v1(nn.Module):
    def __init__(self, input_size, output_size, num_blocks, seq_len=13, num_heads=2, dropout=0.2):
        self.num_blocks = num_blocks
        super(transformer_v1, self).__init__()
        self.encoders_blocks = nn.ModuleList()
        self.layer_normal = nn.LayerNorm(int(input_size * seq_len / 10))

        for i in range(self.num_blocks):
            self.encoders_blocks.append(transformer_encoder_block_v2(input_size=input_size,output_size=input_size,num_heads=2,dropout=dropout,number=i))

        self.ffn_layer1=FFN(input_size=input_size*seq_len, final_size=int(input_size*seq_len/10))
        self.ffn_layer2 = FFN(input_size=int(input_size*seq_len/10), final_size=output_size,activation=False)
    def forward(self,x):
        x=x.permute(1,0,2)
        for i in range(self.num_blocks):
            #print(x.shape)
            x,att_weights=self.encoders_blocks[i](x)
        #print(torch.median(x), torch.max(x), torch.min(x))
        x=self.ffn_layer1(x.permute(1, 0, 2).flatten(start_dim=1))
        x=self.layer_normal(x)
        #print(torch.median(x), torch.max(x), torch.min(x))
        x = self.ffn_layer2(x)
        #print(torch.median(x),torch.max(x),torch.min(x))
        return F.sigmoid(x.flatten())
class simple_attention_block(simple_attention_block):
    def forward(self, x):
        x = x[:, -1:, :]

        #x = self.ffn0(x)
        x = x.permute(2, 0, 1)
        #x[:, -1:, :] = 0

        # this is mention in multihead attention pytorch document
        att_output, att_weight = self.multi_att(key=x, query=x, value=x)

        x = self.layer_normal1(att_output.permute(1, 2, 0) + x.permute(1, 2, 0))
        # print(torch.max(x),torch.min(x),torch.std(x))
        x = F.relu(x)
        x = self.ffn1(x.flatten(start_dim=1))
        x = self.ffn2(x)
        for i in range(att_weight.shape[0]):
            att_weight_ = att_weight[i].detach().numpy()
            # att_weight_ = (att_weight_-np.mean(att_weight_) )/ np.std(att_weight_)
            vmax, vmin = np.max(att_weight_), np.min(att_weight_)
            fig = plt.figure()
            plt.imshow(att_weight_, vmin=vmin, vmax=vmax)
            fig.savefig('/home/pooja/PycharmProjects/amex_default_kaggle/data/debugging/att_weight' + str(i) + '__' + str(
                vmin) + '_' + str(vmax) + '.png', bbox_inches='tight')
            plt.close(fig)



        return F.sigmoid(x.flatten())

class var_attention_block(var_attention_block):
    def forward(self, x):
            # print(torch.max(self.state_dict()['multi_att.in_proj_weight']), torch.min(self.state_dict()['multi_att.in_proj_weight']))
            # x = self.layer_normal0(x) # removing it is causing  slow or almost no learnign
            x = x[:, -1:, :]
            s = self.attn1.ffn2(self.attn1.ffn1(x))
            # x = self.ffn0(x)
            s = s.reshape(x.shape[0], x.shape[2], x.shape[2])
            # s=F.sigmoid(s)
            att_weight = self.attn1.soft(s).permute(0,2,1)


            for i in range(att_weight.shape[0]):
                att_weight_ = att_weight[i].detach().numpy()
                # att_weight_ = (att_weight_-np.mean(att_weight_) )/ np.std(att_weight_)
                vmax, vmin = np.max(att_weight_), np.min(att_weight_)
                fig = plt.figure()
                plt.imshow(att_weight_, vmin=vmin, vmax=vmax)
                fig.savefig(
                    '/home/pooja/PycharmProjects/amex_default_kaggle/data/debugging/att_weight' + str(i) + '__' + str(
                        vmin) + '_' + str(vmax) + '.png', bbox_inches='tight')
                plt.close(fig)
            #s = x * s


            #
            # x = self.layer_normal1(s )
            # # print(torch.max(x),torch.min(x),torch.std(x))
            # x = F.relu(
            #     x)  # relu lets it concentrate on few things only, and this thing worked to establish materilaity of attention block
            #
            # x = self.ffn1(x.flatten(start_dim=1))
            # x = self.ffn2(x)
            # return F.sigmoid(x.flatten())

if __name__ == "__main__":
    model = var_attention_block
    model = model(input_size=24, output_size=1, num_blocks=1,num_heads=1,dropout=0)
    filename = 'transformer_v1_18.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.weight_loc+filename, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    def breakinUsers(r):
        r = r.drop(['customer_ID','target'], axis=1)
        return r.to_numpy()
    key = 'customer_ID'
    target = 'target'
    choosen=['0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fbac11a8ed792feb62a','0001337ded4e1c2539d1a78ff44a457bd4a95caa55ba1730b2849b92ea687f9e','000962b331f602203d5b5fb41b915893e74db55534182d4968a3423c19258584','0008c2f297e1b00bf567c0d2c25f3e3b356f9a3088d2bf47aaaa724d26df8787','00057576e6eab4633ec2893ca7e0ab76f2094ad2d43b1e3749db49d51e064ee9']
    train = pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/data/intermediate_data/dev.csv',
                        nrows=1000, header=0)
    #choosen=list(train['customer_ID'])
    train=train[train.customer_ID.isin(choosen)].groupby(key).tail(1)
    # train = train.set_index(key).select_dtypes(
    #     exclude=['object', 'O']).reset_index()  # todo check for better replacement
    train = train.replace([np.inf, -np.inf, np.nan, np.inf], 0.00).drop('Unnamed: 0',axis=1)
    if train.shape[0] > 1:
        output_dict = {'customer_ID': [], 'prediction': [], 'target': []}
        group = train.groupby('customer_ID').apply(breakinUsers).to_dict()
        max_seq = 13
        temp = []
        for ke in choosen:
            output_dict['customer_ID'].append(ke)
            tuparray = torch.from_numpy(group[ke].astype(np.float))
            seq_len = tuparray.shape[0]
            # tuparray = np.array(tup)

            if seq_len >= max_seq+10000:
                outs = tuparray[-max_seq:, :]
            else:
                seq_len=1
                outs = torch.cat(( torch.zeros((max_seq - seq_len, tuparray.shape[1])),tuparray[-seq_len:,:]))
            if outs[-1][0] > 0:
                outs[-1, 1:12] = torch.rand((11,))-0.5
            else:
                outs[-1, -12:] = torch.rand((12,))-0.5
            temp.append(outs.to(torch.float32))

            # if target_available:
            #     output_dict['target'].append(target_dict[ke])
            # else:
            #     output_dict['target'].append(0)
        outs = torch.stack(temp, dim=0)
        output_dict['prediction'] = model(outs).detach().tolist()
        d=0