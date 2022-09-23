#This code covers gradient level information so that user can get a information on gradients
import numpy as np

class grad_information(object):
    @staticmethod
    def get_grads(model) :#changes with choosen model toDo:change this to gnerelize ones
        list_=[]
        def select_obj(key):
            if key=='ffns':obj=model.ffns
            elif key=='ffns_1':obj=model.ffns_1
            elif key=='ffns2':obj=model.ffns2
            return obj
        for block in model.state_dict().keys():
            obj=select_obj(block.split('.')[0])
            key = int(block.split('.')[1].split(".")[0])
            list_.append(obj[key].lr1.weight.grad.numpy().flatten())
        return list_

    @staticmethod
    def zeros_non_zeros(list_:list,threshold:float=0.00000000000001):
        temp=np.concatenate(list_,axis=0)
        temp=abs(temp)
        size=len(temp)
        temp_zero=temp[(temp<=threshold)]
        temp_non_zero=temp[(temp>threshold)]
        return temp_zero,temp_non_zero,size



    @staticmethod
    def get_quantiles_for_non_zeros(input_:np.array):
        quantiles=[]
        for q in [0.20,0.5,0.8]:
            quantiles.append(np.quantile(input_,q))
        return quantiles
    def do_everything(self,state):
        x, y = state.batch['image_pixels'], state.batch['targets']
        model = state.model
        y_hat = model(x)
        loss = state.criterion(y_hat, y)
        loss.backward()
        grads=self.get_grads(model)
        zeros,non_zeros,size=self.zeros_non_zeros(grads)
        perc_zeros=len(zeros)/size
        quantiles=self.get_quantiles_for_non_zeros(non_zeros)
        return perc_zeros,quantiles



