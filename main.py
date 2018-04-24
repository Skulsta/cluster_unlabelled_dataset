from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

import pandas as pd

with open('seeds_dataset.txt') as f:
    table = pd.read_table(f, sep='-', index_col=0, header=None, names=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                          delimiter='\t|\t\t')

# dataset = pd.read_table('seeds_dataset.txt')

print(table)
