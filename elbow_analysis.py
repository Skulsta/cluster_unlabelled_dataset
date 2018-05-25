from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import gaussian_mixture as gmm
import pandas as pd
import matplotlib.pyplot as plt

colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
            'lengthOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

numpy_array = data.values

y = numpy_array[:, 7]  # The last column. The class label. Not used in our unsupervised learning.
X = numpy_array[:, :7]  # From index 0 to 6. 7 feature types.

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cluster_range = range(1, 7)
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans(num_clusters)
    clusters.fit(X_scaled)
    cluster_errors.append(clusters.inertia_)

clusters_df = pd.DataFrame({"num_clusters":cluster_range, "cluster_errors": cluster_errors})

print(clusters_df[0:10])

plt.figure(figsize=(12,6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
plt.show()
