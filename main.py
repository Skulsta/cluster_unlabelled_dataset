from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

import pandas as pd

dataset = pd.read_table('seeds_dataset.txt')

print(dataset)

print(type(dataset))

dataset = dataset.values

vec = DictVectorizer()

print(type(dataset))


url = "C:\\Users\\Magnus\\Documents\\Universitetsarbeid\\INFO284\\Oblig2\\seedsANSI.txt"

colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient', 'lenghtOfKernelGrove']

data = pd.read_table(url, index_col=None, names=colnames, header=None)


print(data)
print(data.columns)
print(data.shape)