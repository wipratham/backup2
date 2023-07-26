import argparse
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.cluster_utils import read_projector_config, reduce_dimensionality, print_image_clusters, visualize_clusters, copy_images_to_clusters

class ImageCluster:
    """
    A class for clustering images using Gaussian Mixture Model (GMM).

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

    def perform_clustering(self, normalized_features, image_paths, optimal_num_clusters):
        """
        Perform clustering using Gaussian Mixture Model (GMM) with the optimal number of clusters.

        Args:
            normalized_features (numpy.ndarray): The normalized feature vectors for clustering.
            image_paths (list): List of image paths to be clustered.
            optimal_num_clusters (int): The optimal number of clusters determined by the highest Silhouette score.

        Returns:
            numpy.ndarray: Cluster labels assigned to each image in the source_folder.
        """
        gmm = GaussianMixture(n_components=optimal_num_clusters, random_state=0)
        cluster_labels = gmm.fit_predict(normalized_features)

        for label in np.unique(cluster_labels):
            if label == -1:
                continue
            cluster_folder = f"cluster{label}"
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)

        copy_images_to_clusters(self.source_folder, image_paths, cluster_labels)


        return cluster_labels

    def process_image_clustering(self):
        """
        Process image clustering using Gaussian Mixture Model (GMM).

        Reads the projector_config.pbtxt file, performs clustering, and visualizes the clusters.
        """
        normalized_features, image_paths = read_projector_config(self.runs_folder)
        reduced_features = reduce_dimensionality(normalized_features)

        silhouette_scores = self.compute_silhouette_scores(normalized_features, self.max_clusters)
        optimal_num_clusters = self.find_optimal_clusters(silhouette_scores)

        cluster_labels = self.perform_clustering(normalized_features, image_paths, optimal_num_clusters)

        visualize_clusters(reduced_features, cluster_labels)
        print_image_clusters(image_paths, cluster_labels)

    def compute_silhouette_scores(self, normalized_features, max_clusters):
        """
        Compute the silhouette scores for different numbers of clusters.

        Args:
            normalized_features (numpy.ndarray): The normalized feature vectors for clustering.
            max_clusters (int): Maximum number of clusters to consider.

        Returns:
            list: List of silhouette scores for different numbers of clusters.
        """
        silhouette_scores = []
        for num_clusters in range(2, max_clusters + 1):
            gmm = GaussianMixture(n_components=num_clusters, random_state=0)
            cluster_labels = gmm.fit_predict(normalized_features)
            silhouette_scores.append(self.silhouette_score(normalized_features, cluster_labels))
        return silhouette_scores

    def silhouette_score(self, normalized_features, cluster_labels):
        """
        Compute the silhouette score for clustering evaluation.

        Args:
            normalized_features (numpy.ndarray): The normalized feature vectors for clustering.
            cluster_labels (numpy.ndarray): Cluster labels assigned to each image.

        Returns:
            float: The silhouette score for the clustering.
        """
        return np.mean(silhouette_score(normalized_features, cluster_labels))

    def find_optimal_clusters(self, silhouette_scores):
        """
        Find the optimal number of clusters based on the highest Silhouette score.

        Args:
            silhouette_scores (list): List of silhouette scores for different numbers of clusters.

        Returns:
            int: The optimal number of clusters.
        """
        return np.argmax(silhouette_scores) + 2  # Add 2 to start from 2 clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Clustering using Gaussian Mixture Model.")
    parser.add_argument("runs_folder", type=str, help="Path to the runs folder containing the tensors.tsv files.")
    parser.add_argument("max_clusters", type=int, help="Maximum number of clusters to consider.")
    parser.add_argument("source_folder", type=str, help="Path to the folder containing the images.")
    args = parser.parse_args()

    image_cluster = ImageCluster(args.runs_folder, args.max_clusters, args.source_folder)
    image_cluster.process_image_clustering()
