import os
from abc import abstractmethod,ABC
#from input_diagnostics.the_iterator import transformer_slicer,transformer_woe_slicer,abstract_data_transformer
import pandas as pd
import time
import shutil
from input_diagnostics.iv import IV
Nrows=2000000
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
        final_dev = pd.read_csv(self.dev_loc, usecols=[self.var, self.target, self.pk],nrows=Nrows)
        final_val = pd.read_csv(self.val_loc, usecols=[self.var, self.target, self.pk],nrows=Nrows)
        low,high=-1,-1
        if self.outlier_cleaning==1:
            low, high = final_dev[self.var].quantile([0.005, 0.995])
            if self.store_file is None:
                final_dev[self.var]=final_dev[self.var].clip(lower=low,upper=high)
                final_val[self.var] = final_val[self.var].clip(lower=low, upper=high)
        #FOR STANDARDISATION
        rf = pd.read_csv(config.output_loc + 'eda/radar_features.csv', index_col='varName')
        dict = rf[['mean', 'std']].to_dict()

        if self.missing_imputation==1:replacer= final_dev[self.var].mean()
        elif self.missing_imputation == 2: replacer = final_dev[self.var].min()
        elif self.missing_imputation == 3:replacer = final_dev[self.var].min()-1
        elif self.missing_imputation == 0:replacer = 0
        elif self.missing_imputation == 4:
            for file in [final_dev, final_val]:
                file['missing' + self.var] = file[self.var].isna().map(int)
                replacer = 0
        elif self.missing_imputation == 5:
            missing_rep_dic = \
            pd.read_csv('/home/pooja/PycharmProjects/amex_default_kaggle/planning/missing_imput.csv', index_col=['var']) \
                .to_dict()['decisioning']
            replacer = (missing_rep_dic[self.var]-dict['mean'][self.var])/dict['std'][self.var]
        if self.store_file is not None:
            pd.DataFrame({'low':[low],'high':[high],'missing':[replacer]}).to_csv(self.store_file + 'slicer|' + str(-1) + '.csv')
            return None

        else:


            final_dev[self.var]=(final_dev[self.var]-dict['mean'][self.var])/dict['std'][self.var]
            final_val[self.var] = (final_val[self.var] -dict['mean'][self.var])/dict['std'][self.var]
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
            param_file=pd.read_csv(param_file_loc+'/'+file,nrows=Nrows)
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
        train=pd.read_csv(self.dev_loc,usecols=[self.var,self.target,self.pk],nrows=Nrows)
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
            data = pd.read_csv(data_str, usecols=[self.var, self.target, self.pk],nrows=Nrows)
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
        self.varlist=None


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
            df=pd.read_csv(input_df, usecols=[key],nrows=1)

            if self.varlist==None:self.varlist==list(df.columns)
            pd.DataFrame(columns=self.varlist).to_csv(self.output_loc,index=False)

            df = pd.read_csv(input_df, usecols=[key])
            df=df[[key]]
            max_row = df.shape[0]
            rows = df[[key]].drop_duplicates(subset=['customer_ID'], keep='last').index.to_list()
            df, df1 = 0, 0  # for memmory efficiency
            loop = 1
            from_row = 0
            # max_row = rows[-1] + 1
            pred_rows=self.batch


            while from_row <= max_row:
                if from_row == max_row:
                    break
                elif loop * pred_rows <= len(rows):
                    to_row = rows[loop * pred_rows] + 1
                else:
                    to_row = max_row

                df=pd.read_csv(input_df,skiprows=range(1, from_row + 1),
                        nrows=to_row - from_row, header=0,usecols=self.varlist)
                df1=self._apply_transformation(df,param_file_loc=self.var_file_loc)
                df1[self.varlist].to_csv(self.output_loc,mode='a',header=False,index=False)


                loop += 1
                from_row = to_row
        output_df=pd.read_csv(self.output_loc,usecols=['customer_ID'])
        print('rows written {} time taken {}'.format(output_df.shape[0], time.time() - start_time))

class missing_imputation(composite_transformer):
    def __init__(self,missing_rep_dict:dict,mean_std_dict:dict,output_loc:str,batch:int=5000,standardize:int=1,all_var=None):
        """
        :param tranformation: variable -->(transformation,parameters)
        :param loc: location from where transformations will be picked(generate_transformation_file output)
        """
        self.all_var=all_var
        self.dict=missing_rep_dict
        self.mean_std_dict=mean_std_dict
        self.output_loc=output_loc
        self.batch=batch
        self.standardize=standardize
        self.var_file_loc=None

    def _apply_transformation(self,df,param_file_loc=None):
        for v in self.all_var:
            if v in self.dict.keys():
                df[v]=df[v].fillna(self.dict[v])
            if self.standardize:
                #print(v)
                df[v]=(df[v]-self.mean_std_dict['mean'][v])/self.mean_std_dict['std'][v]
                df[v]=df[v].astype(np.float32)
        var_list=['customer_ID']+self.all_var
        if 'target' in df.columns:var_list=var_list+['target']
        df=df[var_list]
        return df
class woe_transformation(composite_transformer):
    def __init__(self,output_loc:str="",batch:int=50000,varlist=None,woe_file=str):
        """
        :param tranformation: variable -->(transformation,parameters)
        :param loc: location from where transformations will be picked(generate_transformation_file output)
        """
        self.output_loc=output_loc
        self.batch=batch
        a = IV(verbose=0)
        a.load(woe_file)
        a.excludeList = ['customer_ID', 'target']
        self.var_file_loc=a
        self.varlist=varlist

    def _apply_transformation(self,df,param_file_loc=None):

        converted = param_file_loc.convertToWoe(df)
        converted['customer_ID'] = df['customer_ID']
        if 'target' in df.columns:converted['target'] = df['target']

        return converted

if __name__ == '__main__':
    from common import *
    trans_df=config.rough_loc+'auc'+'_transformation_winners.csv'
    input_df_loc=config.data_loc + 'from_radar/' + 'dev.csv'
    output_loc=config.weight_loc+'transformations/' +'tr1/'
    final_output_loc=config.data_loc+'intermediate_data/transformed_data/hold_out.csv'
    #composite_transformer.generate_transformation_file(transformation_df=trans_df,input_df_loc=input_df_loc,output_loc=output_loc)
    object=composite_transformer(transformation_df=trans_df,var_file_loc=output_loc,output_loc=final_output_loc)
    object(input_df=config.data_loc +'from_radar/hold_out.csv')
