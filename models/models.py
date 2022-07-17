import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

#from multibox_loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(6, 6),
                 stride=(1, 1), padding=(5, 5), pool_size=(2, 2)):
        super().__init__()
        self.pool_size = pool_size
        self.in_channels = in_channels


        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(np.array(kernel_size) + np.array([0, 0])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 0])),
            bias=False)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input1, pool_size=None, pool_type='max'):
        if pool_size is None: pool_size = self.pool_size
        x = input1

        x = F.relu_(self.conv(input1))
        # x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
            return x




class FeatureExtractor(nn.Module):
    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        super().__init__()
        self.num_blocks = len(channels)
        if self.num_blocks>0 :self.start_channel = channels[0]
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = input_image_dim
        self.fc1_p = fc1_p
        self.mode_train = 1
        self.activation_l = torch.nn.ReLU()
        self.activation = torch.nn.Softmax(dim=1)
        last_channel = start_channel
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim()
        if self.fc1_p[0] is not None:
            self.fc1 = nn.Linear(conv_dim[0], fc1_p[0], bias=True)
            self.fc2 = nn.Linear(fc1_p[0], fc1_p[1], bias=True)
        else :
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=fc1_p[1],
                                              kernel_size=(1, 1), stride=(1, 1),
                                              pool_size=(conv_dim[1][-2],conv_dim[1][-1]), padding=0))
            self.num_blocks+=1



        self.init_weight()
        self.dropout = nn.Dropout(0.3)

    def get_conv_output_dim(self):
        input_ = torch.Tensor(np.zeros((1,1)+self.input_image_dim))
        x = self.cnn_feature_extractor(input_)
        return len(x.flatten()),x.shape

    @staticmethod
    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight*10)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_blocks):
            self.init_layer(self.conv_blocks[i].conv)

        if self.fc1_p[0] is not None:
            self.init_layer(self.fc1)
            self.init_layer(self.fc2)
        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)

    def cnn_feature_extractor(self, x):
        # input 501*64
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
            # x_70=torch.quantile(x, 0.7)
            # x_50 = torch.quantile(x, 1)
            # x=self.activation_l((x-x_50-1)/max(x_50,0.01))
            x = self.activation_l(x)

        return x

    def forward(self, input_):
        x = self.cnn_feature_extractor(input_ / 255)
        if self.fc1_p is None:
            x = x.flatten(start_dim=1, end_dim=-1)
            return self.activation(x)

        x = x.flatten(start_dim=1, end_dim=-1)
        if self.fc1_p[0] is not None:
            x = self.activation_l(x)
            x = self.activation_l(self.fc1(x))
            if self.mode_train == 1:
                x = self.dropout(x)
            x = x / torch.max(x)
            x = self.fc2(x)
        x = self.activation(x)

        return x

    def forward1(self, input_):
        x = self.cnn_feature_extractor(input_)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.activation(x / torch.max(x))

        return x


class FFN(nn.Module):
    def __init__(self, state_size=200,final_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, final_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)
class transformer_encoder_block():
    def __init__(self,input_size,output_size,num_heads=4,drop_out=0.2):
        self.multi_att = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=drop_out)
        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(input_size)
        self.ffn = FFN(input_size,output_size)

    def forward(self, x, question_ids, *args):
        see = False
        outs = []
        loop = 0
        for i in args:
            if loop < 4:
                outs.append(i.unsqueeze(2))
            else:
                outs.append(i)
            loop += 1


        device = x.device

        if see: print(x.shape, question_ids.shape, 23)
        elif False:
            x = torch.cat(
                (x, z,), axis=2)
            # print(x.shape,z.shape)

        #         else:
        #             x = torch.cat(
        #                 (x,outs[0],outs[1],outs[2],outs[3],), axis=2)#xself.pos_embedding(x)
        # print(x[0][0])
        # print(x.shape)
        # x = x + pos_x

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)



        att_output, att_weight = self.multi_att(x, x, x, attn_mask=att_mask)
        if see: print(att_output.shape, z.shape)

        # print(x[0][0])
        if see: print(att_output.shape, z.shape, 1)
        att_output = self.layer_normal(att_output + e)  # +additona+ e
        if see: print(att_output.shape, z.shape, 2)
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        if see: print(att_output.shape, 3)
        if False:
            att_output = torch.cat(
                (att_output, y,), axis=2)
        if see: print(y[0], 12)  # ,att_output)
        x = self.ffn(att_output)
        if see: print(att_output.shape, x.shape, 4)
        # x = self.layer_normal2(x +att_output)#x
        x = self.pred(x)
        # print(att_weight[0][0])

        return x.squeeze(-1), att_weight
class transformer_v1():
    def __init__(self,input_size,output_size,num_blocks):
        self.encoders_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.encoders_blocks.append(transformer_encoder_block(input_size=input_size,output_size=input_size,num_heads=4,drop_out=0.2))

        self.final_layer=FFN(self, state_size=input_size,final_size=output_size)
    def forward(self,x):
        for i in range(self.num_blocks):
            x=self.encoders_blocks[i](x)
        output=self.final_layer(x)
        return output


