"""
Our demonstration of the kmeans algorithm used on the seeds datasets provided. Info284 - Spring 2018.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def make_kmeans_cluster():
    # Names of the columns in the data.
    colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
                'lengthOfKernelGrove', 'classifier']

    # Processes the data using pandas. With parameters formatting the data.
    data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

    # Assigning the data.values as a numpy array.
    numpy_array = data.values

    Y = numpy_array[:, 7]  # The last column. The class label. Not used in our unsupervised learning.
    X = numpy_array[:, :7]  # From index 0 to 6. 7 feature types.

    # Principal Component Analysis. Used to reduce dimensionality of the data.
    pca = PCA(n_components=2)  # Parameter that determines the number of components you want the data reduced to.
    pca.fit(X)  # Fitting X to the PCA model.
    X = pca.transform(X)  # Fitting X and then reducing the dimensionality.

    # Code used for Factor Analysis. Did not go with this as PCA worked better.
    # fa = FactorAnalysis(n_components=2)
    # fa.fit(X)
    # X = fa.transform(X)

    # Applying Kmeans to X(data).
    kmeans = KMeans(n_clusters=3)  # Parameter determines number of clusters.
    kmeans.fit(X)  # Fitting data to kmeans model.
    print("Labelling through K-means")
    print(kmeans.labels_)  # Printing out the labels that KMeans have calculated.


    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    # plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=70, c='blue')
    plt.title('Kmeans Clustering')
    plt.show()


make_kmeans_cluster()
