import os
from abc import abstractmethod,ABC
from the_iterator import transformer_slicer,transformer_woe_slicer,abstract_data_transformer
import pandas as pd
import time
import shutil
import torch , numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

class Abstract_transformation(ABC):
    @abstractmethod
    def __init__(self,trans_params:list):
        pass
    def __call__(self,data):
        pass
class normalizer(Abstract_transformation):
    def __init__(self,f1:dict):
        self.dict=f1
    def __call__(self,df):
        for v in self.dict['mean'].keys():
            try:
                df[v] = (df[v] - self.dict['mean'][v]) / self.dict['std'][v]
            except:
                w=0
        return df
class composite_transformer(Abstract_transformation):
    @staticmethod
    def transformer_interpretor(code:str,var,target,pk,store_loc="",df_loc="")->abstract_data_transformer:
        code_to_object={'Slicer Woe':transformer_woe_slicer,'Slicer':transformer_slicer}
        splitted=code.split("|")
        transformation=splitted[0]
        param=[]

        splitted.remove('')
        for i in range(1,4):
            if len(splitted)<=i:param.append(-1)
            else: param.append(int(splitted[i].split(":")[1]))
        print(var,transformation,param[0])
        object=code_to_object[transformation](var=var,target=target,pk=pk,f1=param[0],f2=param[1],f3=param[2],dev_loc=df_loc,val_loc_str="",store_file=store_loc)
        return object


    @classmethod
    def generate_transformation_file(cls,transformation_df:pd.DataFrame,input_df_loc:str,output_loc:str):
        """
        Algo:
        0: convert transformation file to dict
        1. iterate over variable
            2. fetches target and varaible and aplly tanformation
            3.creates var folder saves file on that folder

        :param tranformation: variable -->(transformation,parameters)
        :param df: data on which transfromation is applied
        :param loc: location where var folder are saved
        :return:
        """
        start=time.time()
        if isinstance(transformation_df,str): transformation_df=pd.read_csv(transformation_df)
        dict=transformation_df.set_index('variable').to_dict()['transformation']

        for key in dict.keys():
            if os.path.exists(output_loc+'/'+key+'/'):shutil.rmtree(output_loc+'/'+key+'/')
            os.mkdir(output_loc+'/'+key+'/')
            transformation=dict[key]
            object=cls.transformer_interpretor(transformation,var=key,target='target',pk='customer_ID',store_loc=output_loc+'/'+key+'/',df_loc=input_df_loc)
            object.convert()
        print('time taken {}'.format(time.time()-start))



    def __init__(self,transformation_df:pd.DataFrame,var_file_loc:str,output_loc:str,batch:int=5000,standardize:int=1):
        """
        :param tranformation: variable -->(transformation,parameters)
        :param loc: location from where transformations will be picked(generate_transformation_file output)
        """
        if isinstance(transformation_df, str): transformation_df = pd.read_csv(transformation_df)
        self.dict=transformation_df.set_index('variable').to_dict()['transformation']
        self.var_file_loc=var_file_loc
        self.output_loc=output_loc
        self.batch=batch
        self.standardize=standardize


    def _apply_transformation(self,df,param_file_loc):
        """
        the input data shouldnt be very large (24 gb ram 0.5m rows)
        algo:
        1. apply woe transformation
        2.convert outlier and missing in to dictionary
         3.apply missing and outier transformation iteratively on each variable


        :param df: daatset whher transfromatin needs to be applied
        :return: df
        """
        df['index'] = 1
        df['index'] = df.groupby(['customer_ID']).cumsum().reset_index()['index']
        for key in self.dict.keys():
            transformation = self.dict[key]
            object = self.transformer_interpretor(transformation,var=key,target='target',pk='customer_ID')
            output_df=object.transform(df,param_file_loc+key+'/')
            df=output_df
        return output_df.drop('index',axis=1)

    def __call__(self,input_df):
        """
        reads part of file transforms and append to main file
        algo:
        1.get max row
        2. writes ending point of each customer_id in a list as a
        3.makes a emplty data frame where all the future file will be appended
        4. iterate over rows of input df
            5. picks abatch from input loc
            6. applies function output_loc which apends output at


        :param input_loc: source data which needs to be modifed
        :param output_loc: modified dta needs to be saved
        :return: df
        """

        start_time = time.time()

        if not isinstance(input_df,str):
            return self._apply_transformation(input_df,self.var_file_loc) # faster version
        else:
            key = 'customer_ID'
            df = pd.read_csv(input_df, nrows=1)
            pd.DataFrame(columns=df.columns).to_csv(self.output_loc, index=False)
            df = pd.read_csv(input_df, usecols=[key])
            df=df[[key]]
            max_row = df.shape[0]
            rows = df[[key]].drop_duplicates(subset=['customer_ID'], keep='last').index.to_list()
            df, df1 = 0, 0  # for memmory efficiency
            loop = 1
            from_row = 0
            # max_row = rows[-1] + 1
            pred_rows=self.batch

            # pool=Pool(cpu_count())
            output_df=pd.DataFrame()
            while from_row <= max_row:
                if from_row == max_row:
                    break
                elif loop * pred_rows <= len(rows):
                    to_row = rows[loop * pred_rows] + 1
                else:
                    to_row = max_row

                df=pd.read_csv(input_df,skiprows=range(1, from_row + 1),
                        nrows=to_row - from_row, header=0)
                df1=self._apply_transformation(df,param_file_loc=self.var_file_loc)
                #if 'target' in list(df1.columns):df1=df['target']
                v=0
                df1.to_csv(self.output_loc,mode='a',header=False,index=False)


                loop += 1
                from_row = to_row
        output_df=pd.read_csv(self.output_loc,usecols=['customer_ID'])
        print('rows written {} time taken {}'.format(output_df.shape[0], time.time() - start_time))



if __name__ == '__main__':
    from common import *
    trans_df=config.rough_loc+'auc'+'_transformation_winners.csv'
    input_df_loc=config.data_loc + 'from_radar/' + 'dev.csv'
    output_loc=config.weight_loc+'transformations/' +'tr1/'
    final_output_loc=config.data_loc+'intermediate_data/transformed_data/dev.csv'
    #composite_transformer.generate_transformation_file(transformation_df=trans_df,input_df_loc=input_df_loc,output_loc=output_loc)
    object=composite_transformer(transformation_df=trans_df,var_file_loc=output_loc,output_loc=final_output_loc)
    object(input_df=config.data_loc +'from_radar/dev.csv')
