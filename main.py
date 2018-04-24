from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

import pandas as pd

dataset = pd.read_table('seeds_dataset.txt')

print(dataset)

print(type(dataset))

dataset = dataset.values

vec = DictVectorizer()

print(type(dataset))