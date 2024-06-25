import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from fdc.visualize import plotCluster


class Clustering:
    def __init__(self, high_dim, low_dim, visual):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.visual = visual

    def Agglomerative(self, number_of_clusters, affinity, linkage):
        ag_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity=affinity, linkage=linkage)
        clusters = ag_cluster.fit_predict(self.high_dim)
        (values, counts) = np.unique(clusters, return_counts=True)
        self.low_dim['Cluster'] = clusters
    
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="UMAP_0", yName="UMAP_1", stroke=3)

        return self.low_dim.Cluster.to_list(), counts
    
    
    def DBSCAN(self, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(self.high_dim)
        (values, counts) = np.unique(clusters, return_counts=True)
        self.low_dim['Cluster'] = clusters
    
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="UMAP_0", yName="UMAP_1", stroke=3)

        return self.low_dim.Cluster.to_list(), counts
    
    def K_means(self, no_of_clusters):
        kmeans = KMeans(n_clusters=no_of_clusters)
        clusters = kmeans.fit_predict(self.high_dim)
        (values, counts) = np.unique(clusters, return_counts=True)
        self.low_dim['Cluster'] = clusters
    
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="UMAP_0", yName="UMAP_1", stroke=3)

        return self.low_dim.Cluster.to_list(), counts

