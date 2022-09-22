import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.init as init
import math
import numpy as np

#from multibox_loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FFN(nn.Module):
    def __init__(self, input_size=200, final_size=200,activation=True,dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = input_size
        #self.layer_normal = nn.LayerNorm(input_size)
        self.lr1 = nn.Linear(input_size, final_size)
        self.activation=nn.ReLU()# nn.LeakyReLU() #
        self.dropout = nn.Dropout(dropout)
        self.apply_activation=activation
        self.init_layer(self.lr1)

    def forward(self, x):
        x = self.lr1(x)
        if self.apply_activation:x = self.activation(x)

        return self.dropout(x)

    @staticmethod
    def init_layer(layer):
        nn.init.kaiming_uniform(layer.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
        #layer.weight.data.fill_(1)

    # def init_weight(self):
    #     for i in range(self.num_blocks):
    #         self.init_layer(self.conv_blocks[i].conv)
    #
    #     if self.fc1_p[0] is not None:
    #         self.init_layer(self.fc1)
    #         self.init_layer(self.fc2)
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
        self.multi_att = nn.MultiheadAttention(embed_dim=4, num_heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal0 = nn.LayerNorm(input_size*4)
        self.layer_normal1 = nn.LayerNorm(input_size)
        self.ffn0 = FFN(input_size , input_size*3 , dropout=dropout, activation=False)
        self.ffn1 = FFN(input_size*4,input_size*2,dropout=dropout,activation=False)
        self.ffn2 = FFN(input_size*2, output_size,activation=False)


    def forward(self, x):
        #print(torch.max(self.state_dict()['multi_att.in_proj_weight']), torch.min(self.state_dict()['multi_att.in_proj_weight']))
        #x = self.layer_normal0(x) # removing it is causing  slow or almost no learnign
        x=x[:,:,:]


        #x=self.ffn0(x)

        x = x.permute(1, 0, 2)

        # this is mention in multihead attention pytorch document
        att_output,att_weight = self.multi_att(key=x, query=x,value= x)

        x=att_output.permute(1,0,2)+x.permute(1,0,2)
        x=self.layer_normal0(x.flatten(start_dim=1))
        #print(torch.max(x),torch.min(x),torch.std(x))
        x=F.relu(x) # relu lets it concentrate on few things only, and this thing worked to establish materilaity of attention block

        x=self.ffn1(x)
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
        #params=[(input_size*13,input_size*13*2,0,0),(input_size*13*2,100,1,0),(100,output_size,0,0)]#,(13,1,0,0),(1,1,0,0),(1*100,100,0,1),(100,1,0,1)]
        params = [(input_size * 13, input_size * 13 * 2, 0, 0), (input_size * 13 * 2, 100, 1, 0),
                  (100, output_size, 0, 0)]  # ,(13,1,0,0),(1,1,0,0),(1*100,100,0,1),(100,1,0,1)]
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
class time_combination_nn(nn.Module):
    def __init__(self,dropout=0.2,input_size=0,output_size=0):
        super(time_combination_nn, self).__init__()
        self.ffns_0 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffns_1 = nn.ModuleList()
        self.ffns2=nn.ModuleList()

        #self.attention=simple_attention_block(input_size=188,output_size=1,num_heads=1,dropout=0.2,num_blocks=1)
        #for i in range(13): self.ffns_0.append(FFN(188,10, dropout=dropout, activation=False))

        for i in range(input_size):self.ffns.append(FFN(13,20,dropout=dropout, activation=False))

        for i in range(input_size): self.ffns_1.append(FFN(20, 13, dropout=dropout, activation=True))
        self.ffns2.append(FFN(input_size*13,100,dropout=dropout, activation=True))
        self.ffns2.append(FFN(100, 1, dropout=dropout, activation=False))
        #self.reset_parameters()
        #self.layer_normal0 = nn.LayerNorm(input_size )
        #self.layer_normal1 = nn.LayerNorm(input_size*13)
    def forward(self, x):
         #x = self.layer_normal0(x)
         # output2 = torch.zeros((x.shape[0], x.shape[1], 10))
         # for j in range(len(self.ffns_0)):
         #     output2[:,j,:]=self.ffns_0[j](x[:,j,:])
         # x=torch.cat((x,output2),dim=2)
         x = x.permute(0, 2, 1)
         output = torch.zeros((x.shape[0], x.shape[1], 13))
         for j in range(len(self.ffns)):
             output[:,j,:]= self.ffns_1[j](self.ffns[j](x[:,j,:]))
         #x=x+output # res connection
         #              ,x.std(dim=2).reshape((x.shape[0],188,1)),torch.max(x,dim=2)[0].reshape(x.shape[0],188,1),
         #              torch.min(x,dim=2)[0].reshape(x.shape[0],188,1)),dim=2)
         #x=torch.cat((output.flatten(start_dim=1),output2.flatten(start_dim=1)),dim=1)
         x=output.flatten(start_dim=1)
         #x=self.attention(output)

         #adding a attentionlayer
         #x=self.layer_normal1(x.flatten(start_dim=1))
         for j in range(len(self.ffns2)):
              x=self.ffns2[j](x)
         return torch.sigmoid(x.flatten())


        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)

class time_combination_nn_with_variable_mixer(nn.Module):
    def __init__(self,dropout=0.2,input_size=0,output_size=0):
        super(time_combination_nn_with_variable_mixer, self).__init__()
        self.ffns_0 = nn.ModuleList()
        self.ffns_01 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffns_10 = nn.ModuleList()
        self.ffns_11 = nn.ModuleList()
        self.ffns_12 = nn.ModuleList()
        self.ffns2=nn.ModuleList()
        self.ffns_20 = nn.ModuleList()
        self.ffns_21 = nn.ModuleList()

        #variable_mixing
        self.ffns_0.append(FFN(input_size,input_size*2, dropout=dropout, activation=True))

        self.ffns_01.append(FFN( input_size * 2,input_size, dropout=dropout, activation=False))
        # seperate time for mixed variable
        self.ffns_20.append(FFN(13, 30, dropout=dropout, activation=False))
        self.ffns_21.append(FFN(30, 30, dropout=dropout, activation=True))
        #self.ffns_21.activation=nn.Sigmoid()
        # self.ffns_0[0].lr1.weight.requires_grad = False
        # self.ffns_0[0].lr1.bias.requires_grad = False
        # self.ffns_01[0].lr1.weight.requires_grad = False
        # self.ffns_01[0].lr1.bias.requires_grad = False


        #time mixing
        # for i in range(input_size):self.ffns.append(FFN(13,20,dropout=dropout, activation=False))
        # for i in range(input_size): self.ffns_1.append(FFN(20, 13, dropout=dropout, activation=True))
        self.ffns_10.append(FFN(13,30,dropout=dropout, activation=False))
        self.ffns_11.append(FFN(30, 13, dropout=dropout, activation=True))
        self.ffns_12.append(FFN(13, 30, dropout=dropout, activation=False))




        #flatten ffn
        self.ffns2.append(FFN(input_size * 30*2 , 100, dropout=dropout, activation=True))  # True:2 F:1 testing
        #self.ffns2.append(FFN(input_size*30*2,100,dropout=dropout, activation=True))#True:2 F:1 testing
        self.ffns2.append(FFN(100, 1, dropout=dropout, activation=False))
        self.layer_normal0 = nn.LayerNorm(31)
        self.layer_normal1 = nn.LayerNorm(input_size*2)
    def forward(self, x):
         #y=x[:]
         #x = self.layer_normal0(x)
         #output2 = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))

         #v mixing and time ixing together
         output2=self.ffns_01[0](self.ffns_0[0](x[:,:,:])) #+x
         output2=self.ffns_21[0](self.ffns_20[0](output2.permute(0,2,1)))
         #output2=F.normalize(output2,dim=2)
         # fragmented fnns

         #x=y[:]
         #output2=torch.zeros(x.shape[0],x.shape[2],30)

         #x=torch.zeros(x.shape)
         #x=torch.cat((x,output2),dim=2)

         x = x.permute(0, 2, 1)
         #output = torch.zeros((x.shape[0], x.shape[1], 26))
         # for j in range(len(self.ffns)):
         #     output[:,j,:]= self.ffns_1[j](self.ffns[j](x[:,j,:]))
         #time mixing
         output = self.ffns_11[0](self.ffns_10[0](x[:, :, :]))
         output=self.ffns_12[0](output)# skip connection  problem time varibale ixing not workng for varibale mixed
         x = torch.cat((output, output2), dim=1)
         # varibale metrics mixing
         #output=self.layer_normal1(output.permute(0,2,1))
         #output = self.ffns_21[0](self.ffns_20[0](output))

         #x=x+output # res connection
         #              ,x.std(dim=2).reshape((x.shape[0],188,1)),torch.max(x,dim=2)[0].reshape(x.shape[0],188,1),
         #              torch.min(x,dim=2)[0].reshape(x.shape[0],188,1)),dim=2)
         #x=torch.cat((output.flatten(start_dim=1),output2.flatten(start_dim=1)),dim=1)
         x=x.flatten(start_dim=1)
         #x=self.attention(output)

         #adding a attentionlayer
         #x=self.layer_normal1(x.flatten(start_dim=1))
         for j in range(len(self.ffns2)):
              x=self.ffns2[j](x)
         return torch.sigmoid(x.flatten())

class time_combination_nn2(nn.Module):
    def __init__(self, dropout=0.2, input_size=0, output_size=0):
        super(time_combination_nn2, self).__init__()
        self.ffns_00 = nn.ModuleList()
        self.ffns_01 = nn.ModuleList()
        self.ffns_02 = nn.ModuleList()

        self.ffns_20 = nn.ModuleList()
        self.ffns_21 = nn.ModuleList()
        self.ffns_1 = nn.ModuleList()



        # self.attention=simple_attention_block(input_size=188,output_size=1,num_heads=1,dropout=0.2,num_blocks=1)
        # for i in range(13): self.ffns_0.append(FFN(188,10, dropout=dropout, activation=False))
        for i in range(input_size):
            #temp1=nn.ModuleList()
            #temp2=nn.ModuleList()
            temp=nn.ModuleList()
            #for j in range(13):
                #temp1.append(FFN(2, 2, dropout=0, activation=False))
                #temp2.append(FFN(2, 1, dropout=0, activation=True))
            temp.append(FFN(2, 1, dropout=0, activation=False))
            self.ffns_00.append(temp)

        #time mixing
        self.ffns_01.append(FFN(13, 20, dropout=dropout, activation=False))
        self.ffns_02.append(FFN(20, 13, dropout=dropout, activation=True))


        #self.ffns_20.append(FFN(input_size*2, 20, dropout=dropout, activation=True))
        #self.ffns_21.append(FFN(20, 13, dropout=dropout, activation=False))


        self.ffns_1.append(FFN(input_size * 13, 100, dropout=dropout, activation=True))
        self.ffns_1.append(FFN(100, 1, dropout=dropout, activation=False))

        self.layer_normal1 = nn.LayerNorm(input_size * 2)

    def forward(self, x):
        #x=self.layer_normal1(x)
        #output3 = torch.zeros((x.shape[0],13))
        #for j in range(len(self.ffns_20)):
        #output3=self.ffns_21[0](self.ffns_20[0](x[:,-1,:]))
        #x=torch.cat((x,output2),dim=2)
        x = x.permute(0, 2, 1)
        output = torch.zeros((x.shape[0], int(x.shape[1]/2), 13))
        output2 = torch.zeros((x.shape[0], 13,int(x.shape[1]/2)))
        for j in range(int(x.shape[1]/2)):
            temp=x[:, j*2:j*2+2, :].permute(0,2,1)

            #for k in range(13):
            output2[:,:,j]=self.ffns_00[j][0](temp[:,:,:]).flatten(start_dim=1)
            #output[:, j, :] = self.ffns_1[j](self.ffns[j](torch.squeeze(self.ffns_0[j](temp).permute(0,2,1))))
        output[:, :, :] = self.ffns_02[0](self.ffns_01[0](output2.permute(0,2,1)))
        # x=x+output # res connection
        #              ,x.std(dim=2).reshape((x.shape[0],188,1)),torch.max(x,dim=2)[0].reshape(x.shape[0],188,1),
        #              torch.min(x,dim=2)[0].reshape(x.shape[0],188,1)),dim=2)
        # x=torch.cat((output.flatten(start_dim=1),output2.flatten(start_dim=1)),dim=1)
        #output=torch.cat((output.flatten(start_dim=1),output3),dim=1)
        x = output.flatten(start_dim=1)
        # x=self.attention(output)

        # adding a attentionlayer
        #x = self.layer_normal1(x.flatten(start_dim=1))
        for j in range(len(self.ffns_1)):
            x = self.ffns_1[j](x)
        return F.sigmoid(x.flatten())



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



