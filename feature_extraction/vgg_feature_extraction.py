import tensorflow as tf
import tensorboard as tb
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class FeatureExtractor:
    """
    A class to extract features using a pre-trained model (ViT or ResNet50) and visualize them in TensorBoard.

    Attributes:
        model_name (str): Name of the pre-trained model (ViT or ResNet50).
        folder_path (str): Path to the folder containing images for feature extraction.
        log_dir (str): Path to the directory for storing TensorBoard logs.
    """

    def __init__(self, model_name, folder_path, log_dir):
        """
        Initialize the FeatureExtractor class with model name, input folder path, and log directory.

        Args:
            model_name (str): Name of the pre-trained model (ViT or ResNet50).
            folder_path (str): Path to the folder containing images for feature extraction.
            log_dir (str): Path to the directory for storing TensorBoard logs.
        """
        self.model_name = model_name
        self.folder_path = folder_path
        self.log_dir = log_dir

    def load_model(self):
        """
        Load the pre-trained ResNet model and set it in evaluation mode.

        Returns:
           torch.nn.Module: The pre-trained model.
        """
        if self.model_name == "vgg16":
          model = models.resnet50(pretrained=True)
        elif self.model_name == "vgg19":
          model = models.resnet101(pretrained=True)
        
        else:
          raise ValueError("Invalid model_name. Supported models: vgg16,vgg19")

        # Remove the classification head
        model = torch.nn.Sequential(*list(model.children())[:-1])

        model.eval()
        return model


    def extract_features(self, image_path, model):
        """
        Extract features from an image using the provided model.

        Args:
           image_path (str): The path to the input image file.
           model (torch.nn.Module): The pre-trained model.

        Returns:
           torch.Tensor: Extracted features as a 2D tensor.
        """
        image = Image.open(image_path)
        image = transforms.Resize((224, 224))(image)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
          extracted_features = model(image)
        return extracted_features.view(extracted_features.size(0), -1)


    def process_images(self):
        # Create a SummaryWriter for TensorBoard visualization
        writer = SummaryWriter(log_dir=self.log_dir)

        # Load the pretrained model (ResNet50)
        model = self.load_model()

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
            extracted_features = self.extract_features(image_path, model)
            writer.add_embedding(extracted_features, global_step=os.path.basename(image_path))

            # Print progress
            print(f"Processed image {i+1}/{total_images}")

        # Close the SummaryWriter
        writer.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract features using a pre-trained vgg model.")
    parser.add_argument("--model_name", type=str, default="vgg16", help="Name of the pre-trained vgg model.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the directory for storing TensorBoard logs.")
    args = parser.parse_args()

    # Create an instance of the FeatureExtractor class and process the images
    feature_extractor = FeatureExtractor(args.model_name, args.folder_path, args.log_dir)
    feature_extractor.process_images()