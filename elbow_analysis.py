from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Names of the columns in the data.
colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
            'lengthOfKernelGrove', 'classifier']

# Processes the data using pandas. With parameters formatting the data.
data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\s+')

# Assigning the data.values as a numpy array.
numpy_array = data.values

y = numpy_array[:, 7]  # The last column. The class label. Not used in our unsupervised learning.
X = numpy_array[:, :7]  # From index 0 to 6. 7 feature types.

# Principal Component Analysis. Used to reduce dimensionality of the data.
pca = PCA(n_components=2)  # Parameter that determines the number of components you want the data reduced to.
pca.fit(X)  # Fitting X to the PCA model.
X = pca.transform(X)  # Fitting X and then reducing the dimensionality.

# Scaler standardizes the data to a mean of 0 and variance of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fitting the data to the scalar model.

# Cluster information
cluster_range = range(1, 7)  # The range of clusters we shall analyze.
cluster_errors = []  # A simple array to store cluster_errors.

# Applying KMeans to clusters and appends cluster error score onto the cluster error array.
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters)  # Apply KMeans to number of clusters.
    clusters.fit(X_scaled)  # Fit the scaled and PCA processed data to the cluster.
    cluster_errors.append(clusters.inertia_)  # Append "score" to the cluster_errors array.

# Defines cluster_df as a pandas dataframe of number of clusters and cluster errors.
clusters_df = pd.DataFrame({"num_clusters":cluster_range, "cluster_errors": cluster_errors})

# Prints the dataframe.
print(clusters_df[0:10])

# Plot figure with right size, plot the elbow curve, X-axis: num_cluste, Y-axis: Cluster errors.
plt.figure(figsize=(12,6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
plt.show()

# The dataframe should be printed out in a dataframe in console and graph should be plotted.

