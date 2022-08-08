#float d_88 :3642 count and nuniques
#highe cats:D_65,b_5,D_49,d_106:657,273,279,256 iniques S_2 date variables
from trainers.transformer_trainer import train # calling is first as later calling causes problem probaly because of same name of some modules
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

class abstract_data_transformer(ABC):
    @abstractmethod
    def __init__(self,dev_df,v_df):pass
    @abstractmethod
    def convert(self):pass
class transformer_slicer(abstract_data_transformer):
    """
    Slice dataset or single variable, optionally provides option for capping and florring
    """
    def __init__(self,var:str,target:str,pk:str,f1:int=0,f2:int=-1,f3:int=-1,dev_loc:str="",val_loc_str="",store_file:str=None):
        self.var=var
        self.target=target
        self.outlier_cleaning=f1
        self.pk=pk
        self.dev_loc=dev_loc
        self.val_loc=val_loc_str
        self.store_file = store_file
        self.missing_imputation=f2
        self.description='Slicer|outlier:{}|missing:{}|'.format(self.outlier_cleaning,self.missing_imputation)
    def convert(self):
        final_dev = pd.read_csv(self.dev_loc, usecols=[self.var, self.target, self.pk])

        low,high=-1,-1
        if self.outlier_cleaning==1:
            low, high = final_dev[self.var].quantile([0.005, 0.995])
            if self.store_file is None:
                final_dev[self.var]=final_dev[self.var].clip(lower=low,upper=high)
                final_val = pd.read_csv(self.val_loc, usecols=[self.var, self.target, self.pk])
                final_val[self.var] = final_val[self.var].clip(lower=low, upper=high)




        if self.missing_imputation==1:replacer= final_dev[self.var].mean()
        elif self.missing_imputation == 2: replacer = final_dev[self.var].min()
        elif self.missing_imputation == 3:replacer = final_dev[self.var].min()-1
        elif self.missing_imputation == 0:replacer = 0
        if self.store_file is not None:
            pd.DataFrame({'low':[low],'high':[high],'missing':[replacer]}).to_csv(self.store_file + 'slicer|' + str(-1) + '.csv')
            return None

        else:
            final_dev[self.var] = final_dev[self.var].fillna(replacer)
            final_val[self.var] = final_val[self.var].fillna(replacer)



        return final_dev,final_val
    def transform(self,data,param_file_loc:str):
        """
        apply missing , outlier treatment to a variable of dataset
        :param df: dataset which needs to be converted
        :param param_file: Dataframe
        :return:
        """
        files = os.listdir(param_file_loc)
        df = data[[self.var, self.pk]].copy()
        # df['index'] = 1
        # df['index'] = df.groupby([self.pk]).cumsum().reset_index()['index']
        #final_output = pd.DataFrame()
        for file in files:
            param_file=pd.read_csv(param_file_loc+'/'+file)
            low,high,replacer=param_file['low'][0],param_file['high'][0],param_file['missing'][0]
            if low==-1 and high==-1:pass
            else :df[self.var]=df[self.var].clip(low,high)



        if replacer == -1: replacer = 0 # so that nan are 0 only
        df[self.var] = df[self.var].fillna(replacer)



        data[self.var]=df[self.var]
        return data







class transformer_woe_slicer(abstract_data_transformer):
    """
    Slice dataset for single variable and coverted that variable in woe.
    """
    def __init__(self,var:str="",target:str="",pk:str="",f1:int=-1,f2:int=-1,f3:int=-1,dev_loc:str="",val_loc_str="",store_file:str=None):
        self.var=var
        self.target=target
        self.data_consideration=f1
        self.pk=pk
        self.store_file=store_file
        self.dev_loc=dev_loc
        self.val_loc=val_loc_str
        self.description = 'Slicer Woe|data_consideration:{}|'.format(self.data_consideration)
    def _woe_tranformation(self, data_consideration):
        """
        keeps only pk,target and self.var, converts self .var into woe and returns.
        :param data_consideration: int|applying different woe to different time period data
        :return: dataframe|converted variable, pk,target variable
        """
        a = IV(getWoe=1, verbose=1, sort_feature=None)
        #a.excludeList = [self.target, self.pk]
        train=pd.read_csv(self.dev_loc,usecols=[self.var,self.target,self.pk])
        if data_consideration==-1:
            train = train.groupby(self.pk).tail(1).reset_index().drop([self.pk,'index'], axis=1)
        else:
            train['index']=1
            train['index']=train.groupby([self.pk]).cumsum().reset_index()['index']
            train=train[train['index']==data_consideration].drop(['index','customer_ID'],axis=1)
        binned = a.binning(train, 'target', maxobjectFeatures=100, varCatConvert=1, qCut=10)


        a.iv_all(binned,self.target)
        if self.store_file is not None:
            return a
        train=0 # freeing up memmory
        output=[]
        for data_str in [self.dev_loc,self.val_loc]:
            data = pd.read_csv(data_str, usecols=[self.var, self.target, self.pk])
            if data_consideration!=-1:
                data['index'] = 1
                data['index'] = data.groupby([self.pk]).cumsum().reset_index()['index']
                data = data[data['index'] == data_consideration].drop('index', axis=1)
            converted = a.convertToWoe(data.drop(self.pk,axis=1))
            converted['customer_ID'] = data['customer_ID']
            converted['target'] = data['target']
            output.append(converted)

        return output
    def convert(self):
        """
        iteratively call woe data for each time point nd appeds
        :param sort_var:
        :return:
        """
        if self.data_consideration==-1:
            if self.store_file is not None:#saving  the params only
                a = self._woe_tranformation(self.data_consideration)
                a.saveVarcards(self.store_file + 'woe|' + str(-1) + '|')
                return None
            else:
                sort=0
                final_dev,final_val= self._woe_tranformation(self.data_consideration)
        else:
            sort=1
            for i in range(1,self.data_consideration+1):
                if self.store_file is not None:
                    a=self._woe_tranformation(i)
                    a.saveVarcards(self.store_file + 'woe|' + str(i) + '|')
                else:
                    df_dev,df_val=self._woe_tranformation(i)
                    df_dev['index'], df_val['index']=i,i
                    if i==1:final_dev,final_val=df_dev,df_val
                    else:
                        final_dev=final_dev.append(df_dev)
                        final_val=final_val.append(df_val)
        if self.store_file is not None:return  None
        print(final_dev[self.var].nunique(), final_val[self.var].nunique())

        if sort==1:return final_dev.sort_values([self.pk,'index']).drop('index',axis=1),final_val.sort_values([self.pk,'index']).drop('index',axis=1)
        else: return final_dev,final_val
    def transform(self,data,param_file_loc:str):
        """
        for a variable transform the input dataset
        algo:take
        :param file:
        :return:
        """
        files=os.listdir(param_file_loc)
        df=data[[self.var,self.pk,'index']].copy()

        final_output=pd.DataFrame()
        for file in files:
            data_consideration=int(file.split("|")[1])
            if data_consideration!=-1:train = df[df['index'] == data_consideration][[self.var]]#.drop('index', axis=1)
            else :train=df[[self.var]]#.drop('index', axis=1)
            a=IV(verbose=0)

            a.load(loc=param_file_loc,name=file.replace('.csv',''))

            converted = a.convertToWoe(train)
            converted['index']=df['index']
            converted[self.pk]=df[self.pk]
            final_output=final_output.append(converted)

        data[self.var]=data[[self.pk,'index','S_2']].set_index([self.pk,'index']).\
            join(final_output.set_index([self.pk,'index'])).reset_index()[self.var].fillna(0)
        return data








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
            if self.var_dict[v] >=20:
                for data_cosideration in [-1,13]:
                    object = transformer_woe_slicer(var=v, target='target', pk='customer_ID', f1=data_cosideration,
                                            f2='', dev_loc=dev_loc, val_loc_str=val_loc)
                    self.cards.append(iteration_card(var=object.var, transformation=(object, object.description)))
            continue
            for outlier_treatment in [0,1]:#code 1-> num:1 cat:2  10-> 1: not missing 2: miss
                object = transformer_slicer(var=v, target='target', pk='customer_ID', f1=outlier_treatment,
                                            f2=0, dev_loc=dev_loc, val_loc_str=val_loc)
                self.cards.append(iteration_card(var=object.var, transformation=(object, object.description)))
            if self.var_dict[v]> 20:
                for missing_treatment in [1,2,3]:
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
            for model in self.models:
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
                train(model=card.objects[1],data_loader=data_loader, data_loader_v= data_loader_v,loss_func=card.objects[2],callbacks=self.callbacks,pretrained= None,epoch=epoch)

if __name__ == '__main__':
    dev_loc=config.data_loc + 'from_radar/' + 'dev.csv'
    val_loc=config.data_loc + 'from_radar/' + 'hold_out.csv'
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
        #var_list = {'D_48':'num'}
        model = mdl.__dict__[config.model]
        model = model(input_size=1, output_size=1, num_blocks=4, num_heads=1)
        models=[(model,config.model)]
        loss_func = loss.__dict__[config.loss_func]
        loss_funcs=[(loss_func,config.loss_func)]
        epoch=10
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
