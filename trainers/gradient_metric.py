#This code covers gradient level information so that user can get a information on gradients
import numpy as np

class grad_information(object):
    def get_grads(model,identifier):
        list_=[]
        for block in model.state_dict().keys():
            key = int(block.split(identifier)[1].split(".")[0])
            list_.append(model.conv_blocks[key].conv.weight.grad)
        return list_
    def zeros_non_zeros(list_:list,threshold:float=0.000001):
        temp=np.array(list_)
        temp=abs(temp)
        size=len(temp)
        temp_zero=temp[temp<=threshold]
        temp_non_zero=temp[temp<=threshold]
        return temp_zero,temp_non_zero,size
    def perc_zeros(input_:np.array,size):
        return len(input_)/size
    def get_quantiles_for_non_zeros(input_:np.array):
        quantiles=[]
        for q in [0.20,0.5,0.8]:
            quantiles.append(np.quantile(input_,q))
        return quantiles
    def do_everything(self,model):
        grads=self.get_grads(model)
        zeros,non_zeros=self.zeros_non_zeros(grads)
        perc_zeros=self.perc_zeros(zeros)
        quantiles=self.get_quantiles_for_non_zeros(non_zeros)
        return perc_zeros,quantiles



