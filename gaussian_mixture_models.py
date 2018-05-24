import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def make_gaussian_cluster():
    colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
        'lenghtOfKernelGrove', 'classifier']

    data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

    numpy_array = data.values
    y = numpy_array[:, 7]   # The last culomn. The class label.
    X = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

    # To be more in flow with the book and field, we could/should
    # call features 'X' and label 'y'
    # print(X)
    #  print(y)

    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)

    # Finding clusters in the same manner as k-means
    gmm = GaussianMixture(n_components=3, random_state=1, covariance_type='diag').fit(X)
    labels = gmm.predict(X)
    plt.scatter(X[:,0], X[:, 2], c=labels, cmap='viridis')
    plt.title('Gaussian Mixture Model Cluster')
    plt.show()

