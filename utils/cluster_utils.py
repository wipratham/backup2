import numpy as np
from sklearn.decomposition import PCA
import re
import os
import shutil
import matplotlib.pyplot as plt

def read_projector_config(runs_folder):
    """
    Read the projector_config.pbtxt file and extract the tensor paths.

    Args:
        runs_folder (str): Path to the folder containing the projector_config.pbtxt file.

    Returns:
        tuple: A tuple containing the normalized feature vectors and image paths.
    """
    # Read the projector_config.pbtxt file
    with open(os.path.join(runs_folder, "projector_config.pbtxt"), "r") as f:
        content = f.read()

    # Use regular expressions to extract the tensor paths
    pattern = r'tensor_name: "default:(.*?)"\n\s+tensor_path: "(.*?)"'
    matches = re.findall(pattern, content)

    # Process features and images
    features = []
    image_paths = []
    for tensor_name, tensor_path in matches:
        tensor_file = os.path.join(runs_folder, tensor_path)

        # Load the feature vector from the tensors.tsv file
        with open(tensor_file, "r") as tsv_file:
            feature_vector = [float(val) for val in tsv_file.readline().strip().split("\t")]
            features.append(feature_vector)

        # Extract the image path from the tensor name
        image_name = tensor_name.split("/")[-1]
        image_paths.append(image_name)

    # Convert the features list to a numpy array
    features = np.array(features)

    # Normalize the feature vectors
    normalized_features = features / np.linalg.norm(features, axis=1)[:, None]

    return normalized_features, image_paths

def reduce_dimensionality(normalized_features):
    """
    Reduce dimensionality of the feature vectors using PCA.

    Args:
        normalized_features (numpy.ndarray): The normalized feature vectors.

    Returns:
        numpy.ndarray: The reduced feature vectors using PCA with 2 components.
    """
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(normalized_features)
    return reduced_features

def print_image_clusters(image_paths, cluster_labels):
    """
    Print the image paths and their corresponding cluster labels.

    Args:
        image_paths (list): List of image paths.
        cluster_labels (numpy.ndarray): Cluster labels assigned to each image.
    """
    # Print the image paths and their corresponding cluster labels
    for image_path, cluster_label in zip(image_paths, cluster_labels):
        print(f"Image: {image_path} | Cluster: {cluster_label}")

def visualize_clusters(reduced_features, cluster_labels):
    """
    Visualize the clusters in a 2D scatter plot.

    Args:
        reduced_features (numpy.ndarray): The reduced feature vectors using PCA.
        cluster_labels (numpy.ndarray): Cluster labels assigned to each image.
    """
    # Visualize the clusters
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Feature Clusters')
    plt.show()

def copy_images_to_clusters(source_folder, image_paths, cluster_labels):
    """
    Copy images to respective cluster folders.

    Args:
        source_folder (str): Path to the folder containing the original images.
        image_paths (list): List of image paths.
        cluster_labels (numpy.ndarray): Cluster labels assigned to each image.
    """
    for image_path, cluster_label in zip(image_paths, cluster_labels):
        if cluster_label == -1:
            continue
        image_src = os.path.join(source_folder, image_path)
        cluster_folder = f"cluster{cluster_label}"
        image_dest = os.path.join(cluster_folder, image_path)
        shutil.copy(image_src, image_dest)
