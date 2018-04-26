import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
        'lenghtOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

numpy_array = data.values
y = numpy_array[:, 7]   # The last culomn. The class label.
X = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

# Get the number of classifications
n_labels = len(np.unique(y))

# To be more in flow with the book and field, we could/should
# call features 'X' and label 'y'
# print(X)
# print(y)

# Finding clusters in the same manner as k-means
gmm = GaussianMixture(n_components=n_labels).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

# Using a probabilistic model to measure the probability
# that any point belongs to the given cluser. Not sure if this is right tho...
probability = gmm.predict_proba(X)
print("Probability that any point belongs to a given cluster")
print(probability[:5].round(3))

# Visualizing it
size = 50 * probability.max(1) ** 2 # Square emphasizies differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)
plt.show()


# Copypasta
# Using ellipses to help visualize the locations and shapes of the GMM clusteres.
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

gmm = GaussianMixture(n_components=4, random_state=42)
# plot_gmm(gmm, X) Not working - 'GaussianMixture' object has no attribute 'covars_'

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
# plot_gmm(gmm, X) - Still no 'covars_'