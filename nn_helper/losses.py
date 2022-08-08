import torch.nn as nn
import numpy as np
import torch

EPSILON_FP16 = 1e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BCELoss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):

        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        return self.func(pred, actual)
class L2Loss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.MSELoss(reduce=False)#nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):

        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        #pred=(pred-torch.min(pred))/torch.max(pred)
        weights=torch.where(actual < 1, torch.tensor(20.0, dtype=torch.float), actual)

        return torch.mean(self.func(pred, actual)*weights)
class L2Loss_with_penality(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.MSELoss()#nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):

        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        sorted = torch.argsort(pred.detach(), descending=True)
        actual = torch.gather(input=actual, dim=0, index=sorted)
        pred = torch.gather(input=pred, dim=0, index=sorted)
        weights = torch.cumsum(torch.where(actual < 1, torch.tensor(20.0, dtype=torch.float), actual), dim=0)
        threshold_index = torch.argmax(torch.tensor(weights > weights[-1] * 0.04, dtype=torch.long))
        threshold = pred[threshold_index]
        extra_loss=0
        count=0
        for i in range(threshold_index):
            if actual[i]==0 and pred[i]>threshold:
                extra_loss+=-torch.log(1-pred[i]+threshold)
                count+=1



        return self.func(pred, actual)+extra_loss/count
class Weighted_BCELoss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.BCELoss(reduce=False)#nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):

        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        sorted = torch.argsort(pred.detach(), descending=True)
        actual = torch.gather(input=actual, dim=0, index=sorted)
        pred = torch.gather(input=pred, dim=0, index=sorted)
        weights = torch.cumsum(torch.where(actual < 1, torch.tensor(20.0, dtype=torch.float), actual), dim=0)
        threshold_index = torch.argmax(torch.tensor(weights > weights[-1] * 0.04, dtype=torch.long))-1
        threshold=pred[threshold_index]

        weights = torch.ones(pred.shape)

        for i in range(threshold_index):
            if actual[i]==0 and (pred[i]-0.3)>0.00000001:
                if pred[i]-threshold<2.1:weights[i]*=10
                else :weights*=1/(pred[i]-threshold)


        #weights[final_indices]*=1/(pred-threshold)
        #for i in final_indices:weights[i]*=(pred-threshold)
        loss_vector=self.func(pred,actual)


        return torch.mean(loss_vector*weights)
class distance_BCELoss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):

        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        sorted=torch.argsort(pred.detach(), descending=True)
        actual = torch.gather(input=actual, dim=0, index=sorted)
        pred=torch.gather(input=pred,dim=0,index=sorted)
        weights = torch.cumsum(torch.where(actual < 1, torch.tensor(20.0, dtype=torch.float), actual),dim=0)
        threshold_index=torch.argmax(torch.tensor(weights>weights[-1]*0.04,dtype=torch.long))
        threshold=pred[threshold_index]
        loss2=torch.where(actual==1,threshold-pred,pred-threshold)
        loss2=loss2[loss2>0]

        return torch.mean(self.func(pred, actual))*0+ torch.mean(torch.sigmoid(loss2))