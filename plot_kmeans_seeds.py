"""
Our demonstration of the kmeans algorithm used on the seeds datasets provided. Info284 - Spring 2018.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis


def make_kmeans_cluster():
    colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
                'lengthOfKernelGrove', 'classifier']

    data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

    numpy_array = data.values

    Y = numpy_array[:, 7]  # The last column. The class label. Not used in our unsupervised learning.
    X = numpy_array[:, :7]  # From index 0 to 6. 7 feature types.

    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    # fa = FactorAnalysis(n_components=2)
    # fa.fit(X)
    # X = fa.transform(X)

    kmeans = KMeans(n_clusters=3, algorithm="full")
    kmeans.fit(X)
    print(kmeans.labels_)

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    # plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=70, c='blue')
    plt.title('Kmeans Clustering')
    plt.show()



