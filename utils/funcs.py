import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc, random, cv2
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from prettytable import PrettyTable


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



