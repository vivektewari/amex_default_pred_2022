#float d_88 :3642 count and nuniques
#highe cats:D_65,b_5,D_49,d_106:657,273,279,256 iniques S_2 date variables
from trainers.transformer_trainer import train # calling is first as later calling causes problem probaly because of same name of some modules
#train=0# todo fix this
import pandas as pd
import nn_helper.losses as loss
import models.models as mdl
from iv import IV
from common import *
from abc import abstractmethod,ABC
import os
from dataManager import dataObject
from nn_helper.dataLoaders import amex_dataset
from nn_helper.callbacks import IteratorCallback


import copy
Nrows=500000
class iteration_card(dataObject):
    def __init__(self,var,transformation:tuple=None,model:tuple=None,loss_func:tuple=None,metrics:dict= {}):
        self.var=var
        self.objects=[]
        self.description=[]
        self.metrics=metrics
        self.model=model
        self.loss_func=loss_func
        self.transformation=transformation
        self.create_objects()
    def create_objects(self):
        self.objects=[]
        self.description=[]
        for ele in [self.transformation,self.model,self.loss_func]:
            if ele is  None:ele=("","")
            self.objects.append(ele[0])
            self.description.append(ele[1])

    def convert_to_df(self):
        dict=self.get_dict()
        for key in dict.keys():dict[key]=[dict[key]]
        return pd.DataFrame(dict)
    def get_dict(self):
        dict = {'variable': self.var, 'transformation': self.description[0], 'model': self.description[1],
                'loss': self.description[2]}
        dict.update(self.metrics)
        return dict

    classmethod
    def extract_datas(cards_list):
        df=pd.DataFrame()
        for card in cards_list:
            df=df.append(card.convert_to_df())
        return df

def variable_type_generator(df_loc:str,dict_loc:str,ignore_list=['customer_ID']):
    """
    This save a json dictionary of cat  ,binary ,float variable
    categorization_logic:
        cat logic <=50 uniques binary logic=2 uniques float >1000 uniques
        numeric: =>100 uniques
    df: dataframe location for which categorization needs to be done.
    dict_loc: where json will be saved
    :return:None store {variable_name:variable_type} in
    """
    df=pd.read_csv(df_loc,nrows=2)
    var_list=set(df.columns).difference(set(ignore_list))
    dict={}
    df = pd.read_csv(df_loc)
    for v in var_list: #code 1-> num:1 cat:2  10-> 1: not missing 2: missing
        if df[v].nunique()<=50:
            dict[v]=2
        else:dict[v]=1
        if df[v].isnull().any(): dict[v]+=20
        else :dict[v]+=10
    with open(dict_loc, "w") as fp:json.dump(dict, fp)
    return dict






class The_iterator():
    def __init__(self,dev_loc:str,val_loc:str,loc:str,target,var_list:dict,transformations:list=[],loss_funcs:list=[],models:list=[],callbacks=None):
        self.dev_loc=dev_loc
        self.val_loc=val_loc
        self.models=models
        self.tranformations=transformations
        self.cards=[]
        self.loss_funcs=loss_funcs
        self.var_dict=var_list
        self.target = target
        self.logger=[]
        self.loc=config.rough_loc
        self.data_loader_class=amex_dataset


        self.callbacks=callbacks
    def save_cards(self):
        iteration_card.extract_datas(self.cards).to_csv(self.loc + 'iteration_cards.csv',index=False)
    def transformer_generator(self):
        for v in self.var_dict.keys():
            #if v not in ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']:continue
            if self.var_dict[v] >=0:
                for data_cosideration in [-1]:#,13-1
                    object = transformer_woe_slicer(var=v, target='target', pk='customer_ID', f1=data_cosideration,
                                            f2='', dev_loc=dev_loc, val_loc_str=val_loc)
                    self.cards.append(iteration_card(var=object.var, transformation=(object, object.description)))
            #continue

            for outlier_treatment in []:#0,1#code 1-> num:1 cat:2  10-> 1: not missing 2: miss
                object = transformer_slicer(var=v, target='target', pk='customer_ID', f1=outlier_treatment,
                                            f2=0, dev_loc=dev_loc, val_loc_str=val_loc)
                self.cards.append(iteration_card(var=object.var, transformation=(object, object.description)))
            if self.var_dict[v]> 20:
                for missing_treatment in [0,1,4,5]:
                    object = transformer_slicer(var=v, target='target', pk='customer_ID', f1=0,
                                                    f2=missing_treatment, dev_loc=dev_loc, val_loc_str=val_loc)
                    self.cards.append(iteration_card(var=object.var, transformation=(object, object.description)))


    def create_dataloader(self,dev_file,val_file,var_list):
        """
        create dataloaders for data
        :return:
        """
        data_loader=[]
        for file in [dev_file, val_file]:
            output_dict=amex_dataset.create_dict(df_loc=file, key='customer_ID', var_drop=['customer_ID', 'target'],
                                         target='target', batch=10000000, max_row=None, identifier='',var_list=var_list,save_loc=None)

            data_loader.append(amex_dataset(group=output_dict, n_skill=4, max_seq=13, dev=True,pickled=False))

        return data_loader
    def _add_model_n_loss_to_cards(self):
        new_cards=[]

        for card in self.cards:
            #for model in self.models:
                if card.description [0]=='Slicer|outlier:0|missing:4|':model=self.models[1]
                else :model=self.models[0]

                for loss_func in self.loss_funcs:
                    card.model,card.loss_func=model,loss_func
                    card.create_objects()
                    new_cards.append(copy.deepcopy(card))
        self.cards=new_cards


    def iterate(self,epoch=50):
            """
            this loops to all the variable list, apply transformation and records metrics measuring the signal of th variable
            algo:
                1. add model and loss func information to each card
                loops each card
                   1.get dataloader from transformations
                   2.call train which runns the model and with callback adds metrices
            :return:None|itteratively adds information to metric sheet
            """
            self._add_model_n_loss_to_cards()
            self.callbacks[0].set_info_dict(self.cards[0].get_dict())
            self.callbacks[0].set_blank_csv() # seeting blank scv
            for card in self.cards:
                datasets=card.objects[0].convert()
                data_loader,data_loader_v=self.create_dataloader(datasets[0],datasets[1],var_list=[card.var])
                self.callbacks[0].set_info_dict(card.get_dict())
                train(model=card.objects[1],data_loader=data_loader, data_loader_v= data_loader_v,loss_func=card.objects[2],callbacks=self.callbacks,pretrained= None,epoch=epoch,lr=0.5)

if __name__ == '__main__':

    dev_loc=config.data_loc + 'from_radar/original_radar/' + 'dev.csv'
    val_loc=config.data_loc + 'from_radar/original_radar/' + 'hold_out.csv'
    variable='D_48'
    if 0:var_list=variable_type_generator(df_loc=dev_loc, dict_loc= config.rough_loc+'all_var', ignore_list = ['Unnamed: 0','customer_ID','target'])
    if 0:
        if True:object =transformer_slicer(var=variable, target='target', pk='customer_ID', outlier_cleaning=0, missing_imputation=2, dev_loc=dev_loc, val_loc_str=val_loc)
        if True:object =transformer_woe_slicer(var=variable,target='target',pk='customer_ID',data_consideration=13,dev_loc=dev_loc,val_loc_str=val_loc,sort_var=['customer_ID','S_2'])
        dev,val=object.convert()
        dev.to_csv(config.rough_loc+'dev.csv')
        val.to_csv(config.rough_loc+'val.csv')
        distReports(dev, detail=True).to_csv(config.rough_loc+'dist_reports_miss2.csv')
    if 1:
        with open(config.rough_loc+'all_var', "r") as fp:var_list =json.load(fp)
        var_list = {'D_48':22,'S_3':22,'P_2':22,'D_61':22,'D_134':22,'D_142':22,'D_53':22}
        model = mdl.__dict__[config.model]
        model1 = model(input_size=1, output_size=1)
        model2 = model(input_size=2, output_size=1)
        models=[(model1,config.model),(model2,config.model)]
        loss_func = loss.__dict__[config.loss_func]
        loss_funcs=[(loss_func,config.loss_func)]
        epoch=25
        callbacks = [IteratorCallback(metric_loc=config.rough_loc+'metrics.csv',last_epoch=epoch)]
        iter_object=The_iterator(dev_loc=dev_loc,val_loc=val_loc,loc=config.rough_loc,target='target',var_list=var_list,loss_funcs=loss_funcs,models=models,callbacks=callbacks)
        iter_object.transformer_generator()
        iter_object._add_model_n_loss_to_cards()
        if 1: iter_object.save_cards()
        if 1:iter_object.iterate(epoch=epoch)

    #result:
    #plain_slice:Passed|matched distribution of slice dataset with existing one
    # cap_floored_slice:Passed|matched distribution of slice dataset with existing one
    #
