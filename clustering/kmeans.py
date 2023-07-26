import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import utils.cluster_utils as cu 

class ImageCluster:
    """
    A class for clustering images using K-means with the optimal number of clusters determined by the Silhouette score.

    Attributes:
        runs_folder (str): Path to the folder containing the tensors.tsv files generated from a projector_config.pbtxt file.
        max_clusters (int): Maximum number of clusters to consider during clustering.
        source_folder (str): Path to the folder containing the images to be clustered.
    """

    def __init__(self, runs_folder, max_clusters, source_folder):
        """
        Initialize the ImageCluster class with the provided parameters.

        Args:
            runs_folder (str): Path to the folder containing the tensors.tsv files generated from a projector_config.pbtxt file.
            max_clusters (int): Maximum number of clusters to consider during clustering.
            source_folder (str): Path to the folder containing the images to be clustered.
        """
        self.runs_folder = runs_folder
        self.max_clusters = max_clusters
        self.source_folder = source_folder

    def find_optimal_clusters(self, normalized_features):
        """
        Find the optimal number of clusters using the Silhouette score.

        Args:
            normalized_features (numpy.ndarray): The normalized feature vectors for clustering.

        Returns:
            int: The optimal number of clusters determined by the highest Silhouette score.
        """
        silhouette_scores = []
        for num_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters)
            cluster_labels = kmeans.fit_predict(normalized_features)
            silhouette_scores.append(silhouette_score(normalized_features, cluster_labels))

        optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Add 2 to start from 2 clusters
        return optimal_num_clusters

    def perform_clustering(self, normalized_features, image_paths, optimal_num_clusters):
        """
        Perform clustering using K-means with the optimal number of clusters.

        Args:
            normalized_features (numpy.ndarray): The normalized feature vectors for clustering.
            image_paths (list): List of image paths to be clustered.
            optimal_num_clusters (int): The optimal number of clusters determined by the highest Silhouette score.

        Returns:
            numpy.ndarray: Cluster labels assigned to each image in the source_folder.
        """
        kmeans = KMeans(n_clusters=optimal_num_clusters)
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        for i in range(optimal_num_clusters):
            cluster_folder = f"cluster{i+1}"
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)

        for image_path, cluster_label in zip(image_paths, cluster_labels):
            image_src = os.path.join(self.source_folder, image_path)
            cluster_folder = f"cluster{cluster_label+1}"
            image_dest = os.path.join(cluster_folder, image_path)
            shutil.copy(image_src, image_dest)
        
        
        return cluster_labels

    def process_image_clustering(self):
        """
        Process image clustering using K-means.

        Reads the projector_config.pbtxt file, performs clustering, and visualizes the clusters.
        """
        normalized_features, image_paths = cu.read_projector_config(self.runs_folder)
        reduced_features = cu.reduce_dimensionality(normalized_features)

        optimal_num_clusters = self.find_optimal_clusters(normalized_features)
        cluster_labels = self.perform_clustering(normalized_features, image_paths, optimal_num_clusters)

        cu.visualize_clusters(reduced_features, cluster_labels)
        cu.print_image_clusters(image_paths, cluster_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Clustering using Tensor features.")
    parser.add_argument("--runs_folder", type=str, required=True,help="Path to the runs folder containing the tensors.tsv files.")
    parser.add_argument("--max_clusters", type=int, required=True,help="Maximum number of clusters to consider.")
    parser.add_argument("--source_folder", type=str, required=True,help="Path to the folder containing the images.")
    args = parser.parse_args()

    image_cluster = ImageCluster(args.runs_folder, args.max_clusters, args.source_folder)
    normalized_features, image_paths = cu.read_projector_config(args.runs_folder)
    reduced_features = cu.reduce_dimensionality(normalized_features)

    optimal_num_clusters = image_cluster.find_optimal_clusters(normalized_features)
    cluster_labels = image_cluster.perform_clustering(normalized_features, image_paths, optimal_num_clusters)

    cu.visualize_clusters(reduced_features, cluster_labels)
    cu.print_image_clusters(image_paths, cluster_labels)
