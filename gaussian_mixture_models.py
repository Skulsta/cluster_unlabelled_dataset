import pandas as pd
from sklearn.mixture import GaussianMixture

colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
        'lenghtOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

numpy_array = data.values
label = numpy_array[:, 7]   # The last culomn. The class label.
features = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

print(features)
print(label)

gmm = GaussianMixture(n_components=3).fit(features)
result_labels = gmm.predict(features)
