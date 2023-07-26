import argparse
from sklearn.cluster import DBSCAN
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
    A class for clustering images using DBSCAN.

    Attributes:
        runs_folder (str): Path to the folder containing the tensors.tsv files generated from a projector_config.pbtxt file.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        source_folder (str): Path to the folder containing the images to be clustered.
    """

    def __init__(self, runs_folder, eps, min_samples, source_folder):
        """
        Initialize the ImageCluster class with the provided parameters.

        Args:
            runs_folder (str): Path to the folder containing the tensors.tsv files generated from a projector_config.pbtxt file.
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            source_folder (str): Path to the folder containing the images to be clustered.
        """
        self.runs_folder = runs_folder
        self.eps = eps
        self.min_samples = min_samples
        self.source_folder = source_folder

    def perform_clustering(self, normalized_features, image_paths):
        """
        Perform clustering using DBSCAN.

        Args:
            normalized_features (numpy.ndarray): The normalized feature vectors for clustering.
            image_paths (list): List of image paths to be clustered.

        Returns:
            numpy.ndarray: Cluster labels assigned to each image in the source_folder.
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(normalized_features)

        for label in np.unique(cluster_labels):
            if label == -1:
                continue
            cluster_folder = f"cluster{label}"
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)
       
        cu.copy_images_to_clusters(self.source_folder, image_paths, cluster_labels)
        return cluster_labels

    def process_image_clustering(self):
        """
        Process image clustering using DBSCAN.

        Reads the projector_config.pbtxt file, performs clustering, and visualizes the clusters.
        """
        normalized_features, image_paths = cu.read_projector_config(self.runs_folder)
        reduced_features = cu.reduce_dimensionality(normalized_features)

        cluster_labels = self.perform_clustering(normalized_features, image_paths)

        cu.visualize_clusters(reduced_features, cluster_labels)
        cu.print_image_clusters(image_paths, cluster_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Clustering using Tensor features.")
    parser.add_argument("runs_folder", type=str, help="Path to the runs folder containing the tensors.tsv files.")
    parser.add_argument("eps", type=float, help="The maximum distance between two samples for them to be considered as in the same neighborhood.")
    parser.add_argument("min_samples", type=int, help="The number of samples in a neighborhood for a point to be considered as a core point.")
    parser.add_argument("source_folder", type=str, help="Path to the folder containing the images.")
    args = parser.parse_args()

    image_cluster = ImageCluster(args.runs_folder, args.eps, args.min_samples, args.source_folder)
    normalized_features, image_paths = cu.read_projector_config(args.runs_folder)
    reduced_features = cu.reduce_dimensionality(normalized_features)

    cluster_labels = image_cluster.perform_clustering(normalized_features, image_paths)

    cu.visualize_clusters(reduced_features, cluster_labels)
    cu.print_image_clusters(image_paths, cluster_labels)
