import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis


def make_gaussian_cluster():
    # Names of the columns in the data.
    colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
                'lengthOfKernelGrove', 'classifier']

    # Processes the data using pandas. With parameters formatting the data.
    data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

    # Assigning the data.values as a numpy array.
    numpy_array = data.values

    y = numpy_array[:, 7]   # The last column. The class label. Not used in our unsupervised learning.
    X = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

    # Principal Component Analysis. Used to reduce dimensionality of the data.
    pca = PCA(n_components=2)  # Parameter that determines the number of components you want the data reduced to.
    pca.fit(X)  # Fitting X to the PCA model.
    X = pca.transform(X)  # Fitting X and then reducing the dimensionality.

    # fa = FactorAnalysis(n_components=4)
    # fa.fit(X)
    # X = fa.transform(X)

    # Finding clusters in the same manner as k-means
    # Parameters sets num_clusters and sets covariance type to spherical which we found to most effective.
    gmm = GaussianMixture(n_components=3, covariance_type='spherical').fit(X)
    labels = gmm.predict(X)  # Sets the labels to their predicted value based on data X.
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')  # Sets the X-axis and Y-axis equal to features 0 and 1.
    plt.title('GMM clustering')  # Title of plot.
    plt.show()  # Shows plot.

