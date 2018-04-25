import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
        'lenghtOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

numpy_array = data.values
label = numpy_array[:, 7]   # The last culomn. The class label.
features = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

# To be more in flow with the book and field, we could/should
# call features 'X' and label 'y'
# print(features)
# print(label)

# Finding clusters in the same manner as k-means
gmm = GaussianMixture(n_components=3).fit(features)
result_labels = gmm.predict(features)
plt.scatter(features[:, 0], features[:, 1], c=result_labels, s=40, cmap='viridis')
plt.show()

# Using a probabilistic model to measure the probability
# that any point belongs to the given cluser. Not sure if this is right tho...
probability = gmm.predict_proba(features)
print("Probability that any point belongs to a given cluster")
print(probability[:5].round(3))

# Visualizing it
size = 50 * probability.max(1) ** 2 # Square emphasizies differences
plt.scatter(features[:, 0], features[:, 1], c=result_labels, cmap='viridis', s=size)
plt.show()
