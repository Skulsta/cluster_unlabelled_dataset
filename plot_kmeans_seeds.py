"""
Our demonstration of the kmeans algorithm used on the seeds datasets provided. Info284 - Spring 2018.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def make_kmeans_cluster():
    colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
                'lengthOfKernelGrove', 'classifier']

    data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

    numpy_array = data.values

    y = numpy_array[:, 7]  # The last column. The class label. Not used in our unsupervised learning.
    X = numpy_array[:, :7]  # From index 0 to 6. 7 feature types.

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(X);
    X = pca.transform(X);

<<<<<<< HEAD
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print(kmeans.labels_)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=70, c='blue')
    plt.show()
=======
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=70, c='blue')
plt.show()
>>>>>>> e3e3ecd2e4f7ea1fd3c4ddffd894ed722b0dd49a
