import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numbers
import matplotlib.pyplot as plt

def treatment_outlier(df, vars, bottomCaps=None, upperCaps=None, percentile=[0.01, 0.99],specialValue=np.nan):
        """
        :param df: dataframe|Dataset for which outlier treatment been done.
        :param vars: string|Variable for which outlier treatment will be done
        :param bottomCaps:string(method) or number|floored value or method
        :param upperCaps:string(method) or number|capped value or method
        :param percentile:decimal 0 to 1 list(2)|which percecentile value is used for caped or flored
        :return: dataframe|with variable been replaced by outlier treated variable. No variable change
        """

        for i in range(0,len(vars)):
                print(vars[i])
                circuit=[np.nan,np.nan] #upper and lower bound collectid in list
                caps=[bottomCaps,upperCaps]#upper and lower bound parameters in list
                for j in range(0,2):
                        #f=caps[j][i]
                        if caps[j] is not None:
                                if caps[j][i]=='percentile':
                                        percentile1 = df[vars[i]].quantile(percentile[j])
                                        circuit[j]=percentile1
                                elif caps[j][i]in [specialValue,str(specialValue)]:
                                        circuit[j] = np.nan
                                elif caps[j][i]=='nan':
                                        circuit[j]=np.nan
                                else:
                                        circuit[j]=float(caps[j][i])
                print(vars[i], circuit[1])
                if not circuit[0] in [np.nan,float('nan')] :  df[vars[i]][df[vars[i]]<circuit[0]]   =    circuit[0]
                if not circuit[1] in [np.nan,float('nan')] :  df[vars[i]][df[vars[i]] > circuit[1]] =    circuit[1]

        return df



def distReports(df,ivReport=None,detail=False,uniqueVaNum=10,exclude=None):
        if isinstance(df, str):df=pd.read_csv(df)
        if exclude is not None:df=df.drop(exclude,axis=1)
        mis = pd.DataFrame({'varName':df.columns.values,'missing':df.isnull().values.sum(axis=0),'dtypes':df.dtypes},index=df.columns.values)  # new df from existing
        basta = (df.describe()).transpose()

        mis['missing_percent'] = mis['missing'] / df.shape[0]  # new column creation

        final = mis.join(basta) # join using index
        if detail:
                uniques = pd.DataFrame({'nuniques': df.nunique()}, index=df.columns.values)
                final =final.join(uniques)
                #indexSubset=df.shape[0]
                final['uniqueValues'] = final['varName'].apply(lambda x: df[x].unique()[0:uniqueVaNum])


        if ivReport is not None :final=final.join(ivReport)

        return final
import numpy as np
import matplotlib.pyplot as plt
def plotGrabh(df,target,location):
        defaults = df[df[target] == 1]
        objectCols = list(df.select_dtypes(include=['object']).columns)
        allCols = df.columns
        numCols = list(set(allCols) - set(objectCols) - set([target]))

        uniques = pd.DataFrame({'nuniques': df[numCols].nunique()}, index=df[numCols].columns.values)
        numCats=list(uniques[uniques['nuniques']<22].index)
        catCols = objectCols + numCats
        contCols=list(set(allCols)-set(catCols))


        for col in catCols:

                fig = plt.figure()
                temp=df[[col]].fillna('Missing')
                temp2 = defaults[[col]].fillna('Missing')
                ax0 = fig.add_subplot(121)
                temp[col].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                                                     pctdistance=0.9, labeldistance=1.2, radius=1.5)
                ax0 = fig.add_subplot(122)
                temp2[col].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                      pctdistance=0.9, labeldistance=1.2, radius=1)
                plt.savefig(location+col+'.png')
                if True:
                    fig = plt.figure(figsize=(10, 10))
                    sns.countplot(x=col, hue=target, data=df)


                    plt.xlabel('{}'.format(col), size=20, labelpad=15)
                    plt.ylabel('Count', size=20, labelpad=15)
                    plt.tick_params(axis='x', labelsize=20)
                    plt.tick_params(axis='y', labelsize=20)

                    plt.legend(['0', '1'], loc='upper center', prop={'size': 18})
                    plt.title('Feature'.format(col), size=20, y=1.05)
                    plt.savefig(location + col + '_histo.png')
                    fig = plt.figure(figsize=(10, 10))
                    rank = df.groupby(col)[target].mean().reset_index()
                    count = df.groupby(col)[target].count().reset_index()
                    ax = sns.lineplot(data=rank, x=col, y=target)
                    ax2 = ax.twinx()
                    sns.lineplot(data=count, x=col, y=target, ax=ax2, color="r")
                    plt.savefig(location + col + '_mean_volume.png')
        #df[target]=df[target].astype('category')
        for col in contCols:
                fig = plt.figure(figsize=(8, 8))
                x=df[df[target]==0][col]
                d=df[df[target]==1][col]
                sns.kdeplot(df[col], label="all")
                sns.kdeplot(x,  label="0")
                sns.kdeplot(d,  label="1")
                #ax0 = fig.add_subplot(1,3,2)
                plt.savefig(location+col+'.png')
                fig = plt.figure(figsize=(8, 8))
                sns.set(style="whitegrid")
                ax = sns.boxplot(x=col,hue_order=target,data=df)
                #ax0 = fig.add_subplot(3, 3, 2)
                plt.savefig(location + col + '_blot.png')


def corrGraph(df, features,location=None):
        cm = df[features].corr()
        featCount=len(features)
        plt.figure(figsize=(featCount, featCount))
        sns.heatmap(cm, annot=True, cmap='viridis')
        if location is not None: plt.savefig(location + 'corrGraph.png')
        else : plt.show()
def catSplitter(df,var,target):
    df=df[[var,target]]
    df=df.replace(np.nan,'Missing')
    d=df.groupby(var)[target].mean().reset_index()
    e=df.groupby(var)[target].count().reset_index()
    f=d.set_index(var).join(e.set_index(var).rename(columns={target:'count'}))
    f=f.sort_values(by=target)
    return f
def catGrouper(df,varDict):
        """
        :param df: DataFrame|for which variable needs to be grouped
        :param varDict: dictionary|key is variable and values are grouped to be merged
        :return: DataFrame| adding binary variable for different group in varDict and removing original var

        """
        zeroList=[]
        output=df
        for var in varDict.keys():
                vals=varDict[var]
                lCounter=1
                for v in vals:
                    if type(v).__name__=='list':
                        output[var+"___"+str(lCounter)]=df[var].apply(lambda x:1 if x in v else 0)
                        if int(output[var+"___"+str(lCounter)].sum()) == 0: zeroList.append(var+"___"+str(lCounter))
                        lCounter+=1
                    else:
                        output[var+"___"+str(v)]=df[var].apply(lambda x:1 if x==v else 0)
                        if int(output[var + "___" + str(v)].sum())==0:zeroList.append(var + "___" + str(v))
                output=output.drop(var,axis=1)
        print(zeroList)
        return output
def crossTab(df,var1,var2,varCount):
    d=pd.pivot_table(data=df,
                    index=var1,
                    values=varCount,
                    columns=var2,aggfunc=np.sum)
    return d