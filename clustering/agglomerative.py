import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
from utils.cluster_utils import read_projector_config, reduce_dimensionality, visualize_clusters, copy_images_to_clusters, print_image_clusters

class AgglomerativeClusterer:
    """
    A class representing an Agglomerative Clustering algorithm.

    Attributes:
        runs_folder (str): The path to the folder containing the tensors.tsv files.
        num_clusters (int): The number of clusters to create.
        source_folder (str): The path to the source folder containing images.

    Methods:
        perform_clustering(features): Perform Agglomerative Clustering on the input features.
        create_cluster_folders(): Create folders to store the cluster images.
        copy_images_to_clusters(cluster_labels, image_paths): Copy images to respective cluster folders.
        visualize_clusters(reduced_features, cluster_labels): Visualize the clusters in a 2D scatter plot.
        print_image_clusters(image_paths, cluster_labels): Print image paths and their corresponding cluster labels.
    """
    def __init__(self, runs_folder, num_clusters, source_folder):
        """
        Initialize the AgglomerativeClusterer with the provided parameters.

        Args:
            runs_folder (str): The path to the folder containing the tensors.tsv files.
            num_clusters (int): The number of clusters to create.
            source_folder (str): The path to the source folder containing images.
        """
        self.runs_folder = runs_folder
        self.num_clusters = num_clusters
        self.source_folder = source_folder

    def perform_clustering(self, features):
        """
        Perform Agglomerative Clustering on the input features.

        Args:
            features (numpy.ndarray): The input features for clustering.

        Returns:
            numpy.ndarray: The cluster labels assigned to each data point.
        """
        agglomerative = AgglomerativeClustering(n_clusters=self.num_clusters, linkage='ward')
        return agglomerative.fit_predict(features)

    def create_cluster_folders(self):
        """
        Create folders to store the cluster images.
        """
        for i in range(1, self.num_clusters + 1):
            cluster_folder = f"cluster{i}"
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)

    def copy_images_to_clusters(self, cluster_labels, image_paths):
        """
        Copy images to respective cluster folders.

        Args:
            cluster_labels (numpy.ndarray): Cluster labels assigned to each image.
            image_paths (list): List of image paths.
        """
        self.create_cluster_folders()  
        cluster_labels += 1  
        copy_images_to_clusters(self.source_folder, image_paths, cluster_labels)

    def visualize_clusters(self, reduced_features, cluster_labels):
        """
        Visualize the clusters in a 2D scatter plot.

        Args:
            reduced_features (numpy.ndarray): The reduced feature vectors using PCA.
            cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.
        """
        visualize_clusters(reduced_features, cluster_labels)

    def print_image_clusters(self, image_paths, cluster_labels):
        """
        Print the image paths and their corresponding cluster labels.

        Args:
            image_paths (list): List of image paths.
            cluster_labels (numpy.ndarray): Cluster labels assigned to each image.
        """
        print_image_clusters(image_paths, cluster_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster images using feature vectors from the runs folder.")
    parser.add_argument("--runs_folder", type=str, required=True, help="Path to the runs folder containing the tensors.tsv files.")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters.")
    parser.add_argument("--source_folder", type=str, required=True, help="Path to the source folder containing images.")
    args = parser.parse_args()

    # Utilize the functions from cluster_utils.py
    normalized_features, image_paths = read_projector_config(args.runs_folder)
    standardized_features = StandardScaler().fit_transform(normalized_features)
    reduced_features = reduce_dimensionality(normalized_features)

    clusterer = AgglomerativeClusterer(args.runs_folder, args.num_clusters, args.source_folder)

    cluster_labels = clusterer.perform_clustering(standardized_features)

    clusterer.create_cluster_folders()
    clusterer.copy_images_to_clusters(cluster_labels, image_paths)

    clusterer.visualize_clusters(reduced_features, cluster_labels)

    clusterer.print_image_clusters(image_paths, cluster_labels)
