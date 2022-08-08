import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc, random, cv2
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prettytable import PrettyTable
import torch
from sklearn.metrics import roc_curve, auc
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_dict_from_class(class1):
    return {k: v for k, v in class1.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}

class clusterring():
    @classmethod
    def kmeans(cls,x,standardized=False,max_clusters=20,n_cluster=None):
        from sklearn.cluster import KMeans
        import seaborn as sns

        if standardized :
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        if n_cluster is not None:
            kmeans = KMeans( n_clusters=n_cluster, init='k-means++')
            kmeans.fit(x)

            if len(x.shape)<3:
                if standardized :x = scaler.inverse_transform(x)
                frame = pd.DataFrame({'x':x[:,0], 'y':x[:,1],'cluster': kmeans.labels_})
                sns.scatterplot(data=frame, x="x", y="y", hue="cluster")
                plt.show()
                sns.countplot(x="cluster", data=frame)
                plt.show()
            return scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            SSE = []
            for cluster in range(1, max_clusters):
                kmeans = KMeans( n_clusters=cluster, init='k-means++')
                kmeans.fit(x)
                SSE.append(kmeans.inertia_)

            # converting the results into a dataframe and plotting them
            frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})
            plt.figure(figsize=(12, 6))
            plt.plot(frame['Cluster'], frame['SSE'], marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.show()


def lorenzCurve(y_test,y_score,save_loc=None,weight=1):
    n_classes = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _= roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if save_loc is not None:plt.savefig(save_loc)
    else:plt.show()
