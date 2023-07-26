import os
import cv2

def load_image(image_path):
    """
    Load an image from the given path.

    Args:
        image_path (str): The path to the image.

    Returns:
        numpy.ndarray: The loaded image.
    """
    return cv2.imread(image_path)

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.

    Args:
        directory_path (str): The path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)

def save_image(image, output_path):
    """
    Save the image to the specified path.

    Args:
        image (numpy.ndarray): The image to save.
        output_path (str): The path where the image will be saved.
    """
    cv2.imwrite(output_path, image)

