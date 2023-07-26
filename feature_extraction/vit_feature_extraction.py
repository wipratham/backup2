import tensorflow as tf
import tensorboard as tb
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.feature_extraction_utils import load_model, extract_features

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class FeatureExtractor:
    """
    A class to extract features using a pre-trained Vision Transformer model and visualize them in TensorBoard.

    Attributes:
        model_name (str): Name of the pre-trained Vision Transformer model.
        model_path (str): Default model path (leave empty to use the default pre-trained model).
        folder_path (str): Path to the folder containing images for feature extraction.
        log_dir (str): Path to the directory for storing TensorBoard logs.
    """

    def __init__(self, model_name, folder_path, log_dir):
        """
        Initialize the FeatureExtractor class with model name, input folder path, and log directory.

        Args:
            model_name (str): Name of the pre-trained Vision Transformer model.
            folder_path (str): Path to the folder containing images for feature extraction.
            log_dir (str): Path to the directory for storing TensorBoard logs.
        """
        self.model_name = model_name
        self.model_path = ""  # Default model path (leave empty to use the default pre-trained model)
        self.folder_path = folder_path
        self.log_dir = log_dir

    def process_images(self):
        # Create a SummaryWriter for TensorBoard visualization
        writer = SummaryWriter(log_dir=self.log_dir)

        # Load the pretrained ViT model
        model = load_model(self.model_name)

        # List to store the image paths
        image_paths = []

        # Iterate over the images in the folder
        for file_name in os.listdir(self.folder_path):
            image_path = os.path.join(self.folder_path, file_name)

            # Skip directories
            if os.path.isdir(image_path):
                continue

            image_paths.append(image_path)

        # Count the total number of images
        total_images = len(image_paths)

        # Extract features and add to TensorBoard
        for i, image_path in enumerate(image_paths):
            extracted_features = extract_features(image_path, model)
            writer.add_embedding(extracted_features, global_step=os.path.basename(image_path))

            # Print progress
            print(f"Processed image {i+1}/{total_images}")

        # Close the SummaryWriter
        writer.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract features using a pre-trained Vision Transformer model.")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="Name of the pre-trained Vision Transformer model.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the directory for storing TensorBoard logs.")
    args = parser.parse_args()

    # Create an instance of the FeatureExtractor class and process the images
    feature_extractor = FeatureExtractor(args.model_name, args.folder_path, args.log_dir)
    feature_extractor.process_images()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract features using a pre-trained Vision Transformer model.")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="Name of the pre-trained Vision Transformer model.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the directory for storing TensorBoard logs.")
    args = parser.parse_args()

    # Create an instance of the FeatureExtractor class and process the images
    feature_extractor = FeatureExtractor(args.model_name, args.folder_path, args.log_dir)
    feature_extractor.process_images()
