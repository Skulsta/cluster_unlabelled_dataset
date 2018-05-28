
How to run the code:
Run the main.py file, and the two visualizations of K-means and GMM will occur in the Sci-View frame. Doing this,
the specific labelling will be presented in the terminal.
If you add the seed dataset under the file name "seeds_dataset.txt", the main method will run smoothly.
Elsewhere, in the methods "make_gaussian_cluster()" and "make_kmeans_cluster", at the line
data = pd.read_table, change seeds_dataset.txt to an input file of your wish.

If you want to run each algorithms alone, run gaussian_kmeans_models.py and plot_kmeans_seeds.py respectively.

To see how we used Silhouette score analysis and elbow analysis, run silhouette_score_analysis.py
and elbow_analysis.py and their results will occur in the terminal and in the Sci-View frame.





