from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

import pandas as pd

#with open('seeds_dataset.txt') as f:
#    table = pd.read_table(f, sep='-', index_col=0, header=None, names=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
#                          delimiter='\t|\t\t')

# dataset = pd.read_table('seeds_dataset.txt')

# print(table)
# print(type(dataset))

# dataset = dataset.values


colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
            'lenghtOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\t\t|\t')


print(data)
print(data.columns)
print(data.shape)
print(type(data))

numpy_array = data.values
print(type(numpy_array))

# print(numpy_array)
label = numpy_array[:, 7] # The last culomn. The class label.
features = numpy_array[:, :7] # From index 0 to 6. 7 feature types.

print(features)
print(label)
