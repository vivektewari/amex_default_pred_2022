import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

#from multibox_loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FFN(nn.Module):
    def __init__(self, input_size=200, final_size=200,activation=True,dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = input_size
        #self.layer_normal = nn.LayerNorm(input_size)
        self.lr1 = nn.Linear(input_size, final_size)
        self.activation= nn.ReLU()#nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.apply_activation=activation

    def forward(self, x):
        x = self.lr1(x)
        if self.apply_activation:x = self.activation(x)

        return self.dropout(x)
class transformer_encoder_block_v2(nn.Module):
    def __init__(self,input_size,output_size,num_heads=2,dropout=0.2):
        super(transformer_encoder_block_v2, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal0 = nn.LayerNorm(input_size)
        self.layer_normal1 = nn.LayerNorm(input_size)
        self.ffn1 = FFN(input_size,input_size,dropout)
        self.ffn2 = FFN(input_size, input_size,activation=False)


    def forward(self, x):

        #x=self.layer_normal0(x)
        att_output, att_weight = self.multi_att(key=x, query=x,value= x)
        x=self.layer_normal1(att_output+x)
        x=self.ffn1(self.ffn2(x))+x

        return x, att_weight
class transformer_v1(nn.Module):
    def __init__(self,input_size,output_size,num_blocks,seq_len=13,num_heads=2,dropout=0.2):
        self.num_blocks=num_blocks
        super(transformer_v1, self).__init__()
        self.encoders_blocks = nn.ModuleList()
        self.layer_normal = nn.LayerNorm(int(input_size*seq_len/10))

        for i in range(self.num_blocks):
            self.encoders_blocks.append(transformer_encoder_block_v2(input_size=input_size,output_size=input_size,num_heads=num_heads,dropout=dropout))

        self.ffn_layer1=FFN(input_size=input_size*seq_len, final_size=int(input_size*seq_len/10))
        self.ffn_layer2 = FFN(input_size=int(input_size*seq_len/10), final_size=output_size,activation=False)
    def forward(self,x):
        #print(torch.max(x),torch.min(x),torch.mean(x))
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
class transformer_encoder_block(nn.Module):
    def __init__(self,input_size,output_size,num_heads=2,drop_out=0.2):
        super(transformer_encoder_block, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=drop_out)
        self.dropout = nn.Dropout(0.2)

        self.layer_normal = nn.LayerNorm(input_size)
        self.ffn = FFN(input_size,output_size)
        if input_size==output_size:self.residual_connection=True
        else:self.residual_connection=False

    def forward(self, x):
        x = x.permute(1, 0, 2)

        att_output, att_weight = self.multi_att(key=x, query=x,value= x)

        att_output = self.layer_normal(att_output.permute(1,0,2) )  # +additona+ e
        #att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        #print(x.shape)
        x = self.ffn(att_output)



        return x.squeeze(-1), att_weight
class simple_attention_block(nn.Module):
    def __init__(self,input_size,output_size,num_heads=2,dropout=0.2,num_blocks=1):
        super(simple_attention_block, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal0 = nn.LayerNorm(input_size)
        self.layer_normal1 = nn.LayerNorm(input_size)
        self.ffn0 = FFN(input_size , input_size*3 , dropout=dropout, activation=False)
        self.ffn1 = FFN(input_size,input_size*2,dropout=dropout,activation=False)
        self.ffn2 = FFN(input_size*2, output_size,activation=False)


    def forward(self, x):
        #print(torch.max(self.state_dict()['multi_att.in_proj_weight']), torch.min(self.state_dict()['multi_att.in_proj_weight']))
        #x = self.layer_normal0(x) # removing it is causing  slow or almost no learnign
        x=x[:,-1:,:]

        #x=self.ffn0(x)

        x = x.permute(2, 0, 1)

        # this is mention in multihead attention pytorch document
        att_output,att_weight = self.multi_att(key=x, query=x,value= x)

        x=self.layer_normal1(att_output.permute(1,2,0)+x.permute(1,2,0))
        #print(torch.max(x),torch.min(x),torch.std(x))
        x=F.relu(x) # relu lets it concentrate on few things only, and this thing worked to establish materilaity of attention block

        x=self.ffn1(x.flatten(start_dim=1))
        x=self.ffn2(x)
        return F.sigmoid(x.flatten())
class var_attention_module(nn.Module):
    def __init__(self, input_size,  dropout=0.2, num_blocks=1):
        super(var_attention_module, self).__init__()
        self.ffn0 = FFN(input_size, input_size , dropout=dropout, activation=False)
        self.ffn1 = FFN(input_size, input_size * 4, dropout=dropout, activation=True)
        self.ffn2 = FFN(input_size * 4, input_size * input_size, dropout=dropout, activation=False)
        self.soft = nn.Softmax(dim=2)
        self.layer_normal = nn.LayerNorm(input_size )
        self.layer_normal0 = nn.LayerNorm(input_size)

    def forward(self, x):

        s=self.ffn2(self.ffn1(x))
        #x = self.ffn0(x)
        s=s.reshape(x.shape[0],x.shape[2],x.shape[2])
        #s=F.sigmoid(s)
        s=self.soft(s)
        #s=torch.where(s>0.00001,s,torch.tensor(0.0,dtype=torch.float))

        s=torch.bmm(x,s.permute(0,2,1))  # attention
        s=self.layer_normal(s)  # skip connection+0.5*x
        return s

class var_attention_block(nn.Module):
    def __init__(self,input_size,output_size,num_heads=2,dropout=0.2,num_blocks=1):
        super(var_attention_block, self).__init__()
        #self.multi_att = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn1= var_attention_module(input_size,dropout)
        self.attn2=var_attention_module(input_size,dropout)
        self.ffn1 = FFN(input_size , input_size , activation=True)
        self.ffn2 = FFN(input_size , input_size, activation=False)
        self.ffn3 = FFN(input_size, input_size , activation=True)
        self.ffn4 = FFN(input_size , output_size, activation=False)

    def forward(self, x):
        #print(torch.max(self.state_dict()['multi_att.in_proj_weight']), torch.min(self.state_dict()['multi_att.in_proj_weight']))
        #x = self.layer_normal0(x) # removing it is causing  slow or almost no learnign
        x=x[:,-1:,:]
        #x=self.attn1(x)

        #x=self.ffn2(self.ffn1(x))
        #x=self.attn2(x)

        #x=self.ffn3(x)
        x=self.ffn4(x)


        return F.sigmoid(x.flatten())
class simple_nn(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super(simple_nn, self).__init__()
        self.ffns=nn.ModuleList()
        #self.dropout = nn.Dropout(dropout)
        self.layer_norm=nn.ModuleList()
        self.layer_normal0 = nn.LayerNorm(input_size*13)
        self.layer_normal1 = nn.LayerNorm(input_size*13*2)
        self.dropout=dropout
        # featur engenereing in params 0,1 | time extracting variable sin params 2,3, |100 combination from column space
        params=[(input_size*13,input_size*13*2,0,0),(input_size*13*2,100,1,0),(100,output_size,0,0)]#,(13,1,0,0),(1,1,0,0),(1*100,100,0,1),(100,1,0,1)]
        loop=0
        self.params=[]
        for param in params:
            loop += 1
            self.params.append(param)
            if param[3]==1:self.layer_norm.append(nn.LayerNorm(param[0]))
            else:self.layer_norm.append(nn.LayerNorm(0))

            self.ffns.append(FFN(param[0],param[1], dropout=self.dropout, activation=param[2]))



    def forward(self, x):

        for i in range(len(self.ffns)):
            # dim 0-> batch 1->different time 2-> diffrent variable for a time (batch,13,24)
            x=x.flatten(start_dim=1)
            if self.params[i][3] != 0: x = self.layer_norm[i](x)
            x = self.ffns[i](x)





        return F.sigmoid(x.flatten()) #, att_weight
class transformer_v2(nn.Module):
    def __init__(self,input_size,output_size,num_blocks,seq_len=13):
        self.num_blocks=num_blocks
        super(transformer_v2, self).__init__()
        self.encoders_blocks = nn.ModuleList()
        input_size_,output_size_=input_size,input_size
        for i in range(self.num_blocks):
            input_size_ = output_size_
            output_size_ = int(output_size_ / 4) * 2
            #print(input_size_, output_size_)
            self.encoders_blocks.append(transformer_encoder_block(input_size=input_size_,output_size=output_size_,num_heads=2,drop_out=0.2))


        self.final_layer=FFN(input_size=output_size_*seq_len, final_size=output_size)
    def forward(self,x):
        x=x.permute(1,0,2)
        for i in range(self.num_blocks):
            #print(x.shape)
            x,att_weights=self.encoders_blocks[i](x)


        x=self.final_layer(x.permute(1, 0, 2).flatten(start_dim=1))
        return F.sigmoid(x.flatten())
if __name__ == "__main__":
    model=transformer_v1(input_size=188,output_size=1,num_blocks=1)
    out=model(torch.ones((13,10,188)))
    h=0



