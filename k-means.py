import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

import seaborn as sns; sns.set()
import numpy as np



colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
        'lenghtOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

numpy_array = data.values
y = numpy_array[:, 7]   # The last culomn. The class label.
X = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

# Get the number of classifications
n_labels = len(np.unique(y))

# Plot the data with K Means Labels
kmeans = KMeans(n_labels, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()
