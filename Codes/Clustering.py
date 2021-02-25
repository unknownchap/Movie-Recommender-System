import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from Gap import optimalK
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import SilhouetteVisualizer

def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:        
        genre_movies = movies[movies['genres'].str.contains(genre) ]
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)    
    genre_ratings.columns = column_names
    genre_ratings['UserID'] = range(1,611)
    return genre_ratings

# =============================================================================
# Dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
movies.head()

# Import the ratings dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings.head()

genre_ratings = get_genre_ratings(ratings, movies,
                                  ['Romance', 'Sci-Fi', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Thriller'],
                                  ['Romance', 'Sci-Fi', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Thriller'])
genre_ratings = genre_ratings.dropna()
genre_ratings.head()
X = genre_ratings.iloc[:, 0:6]
pca = PCA(n_components=2)
# =============================================================================

# =============================================================================
# Using the Elbow to find the optimal number of clusters
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("Elbow.png")
plt.show()

# =============================================================================

# =============================================================================
# Using the Gap statistic to find the optimal number of clusters
k, gapdf = optimalK(X, nrefs=5, maxClusters = 10)
plt.plot(gapdf.clusterCount, gapdf.gap, linewidth = 3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s = 250, c = 'r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.savefig("Gap Values.png")
plt.show()

# =============================================================================

# =============================================================================
# Using the silhouette to find the optimal number of clusters

for n_clusters in range(4,10):
     model = KMeans(n_clusters, init = 'k-means++')
     cluster_labels = model.fit_predict(X)
     visualizer = SilhouetteVisualizer(model)
     visualizer.fit(X) # Fit the training data to the visualizer
     visualizer.show(outpath="BoW_Silhouette %d" % n_clusters)
     visualizer.poof() # Draw/show/poof the data
     silhouette_avg = silhouette_score(X, cluster_labels)
     print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# =============================================================================

# =============================================================================
# Clustering Using K-Means
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# reduce the features to 2D
reduced_features = pca.fit_transform(X)
# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)


# Visualising the clusters for Train
plt.scatter(reduced_features[y_kmeans == 0, 0], reduced_features[y_kmeans == 0, 1],
            s = 30, c = 'red', label = 'Cluster 1')
plt.scatter(reduced_features[y_kmeans == 1, 0], reduced_features[y_kmeans == 1, 1],
            s = 30, c = 'blue', label = 'Cluster 2')
plt.scatter(reduced_features[y_kmeans == 2, 0], reduced_features[y_kmeans == 2, 1],
            s = 30, c = 'green', label = 'Cluster 3')
plt.scatter(reduced_features[y_kmeans == 3, 0], reduced_features[y_kmeans == 3, 1],
            s = 30, c = 'cyan', label = 'Cluster 4')
plt.scatter(reduced_cluster_centers[:, 0], 
            reduced_cluster_centers[:,1], marker='x', s=200, c='yellow')
plt.title('Clusters of Users by K-Means')
plt.xlabel('Reduced Feature 1')
plt.ylabel('Reduced Feature 2')
plt.legend()
plt.savefig("Users Cluster.png")
plt.show()

ss_kmeans = silhouette_score(X, labels = kmeans.predict(X))

# =============================================================================

# =============================================================================
# Using the Dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(pca.fit_transform(X), method = 'ward'))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.savefig("Dendrogram.png")
plt.show()
# =============================================================================

# =============================================================================
# Clustering Using Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(reduced_features[y_hc == 0, 0], reduced_features[y_hc == 0, 1],
            s = 30, c = 'red', label = 'Cluster 1')
plt.scatter(reduced_features[y_hc == 1, 0], reduced_features[y_hc == 1, 1],
            s = 30, c = 'blue', label = 'Cluster 2')
plt.scatter(reduced_features[y_hc == 2, 0], reduced_features[y_hc == 2, 1],
            s = 30,c = 'green', label = 'Cluster 3')
plt.scatter(reduced_features[y_hc == 3, 0], reduced_features[y_hc == 3, 1],
            s = 30, c = 'cyan', label = 'Cluster 4')

plt.title('Clusters of Users Using Hierarchical Clustering')
plt.xlabel('Reduced Feature 1')
plt.ylabel('Reduced Feature 2')
plt.legend()
plt.savefig("HC.png")
plt.show()


ss_hc = silhouette_score(X, labels = hc.fit_predict(X))


# =============================================================================

