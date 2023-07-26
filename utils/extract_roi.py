import os
import argparse
from image_utils import load_image, create_directory, save_image

class RegionOfInterestExtractor:
    """
    A class to extract region of interest objects from YOLO training dataset images.

    Attributes:
        images_folder (str): Path to the folder containing input images.
        labels_folder (str): Path to the folder containing YOLO detection files.
        output_folder (str): Path to the folder where the extracted region of interest will be saved.
        class_name (str): The class name for which to extract region of interest objects.
        classes_file (str): Path to the file containing class names.
    """

    def __init__(self, images_folder, labels_folder, output_folder, class_name, classes_file):
        """
        Initialize the RegionOfInterestExtractor class with input and output folder paths.

        Args:
            images_folder (str): Path to the folder containing input images.
            labels_folder (str): Path to the folder containing YOLO detection files(.txt).
            output_folder (str): Path to the folder where the extracted region of interest will be saved.
            class_name (str): The class name for which to extract region of interest objects.
            classes_file (str): Path to the file containing class names.
        """
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.output_folder = output_folder
        self.class_name = class_name
        self.classes_file = classes_file

    def create_output_folder(self):
        """
        Create the output folder if it doesn't exist.
        """
        create_directory(self.output_folder)

    def load_image(self, image_file):
        """
        Load an image from the images folder.

        Args:
            image_file (str): The filename of the image.

        Returns:
            numpy.ndarray: The loaded image.
        """
        image_path = os.path.join(self.images_folder, image_file)
        return load_image(image_path)

    def read_yolo_detection_file(self, image_file):
        """
        Read the corresponding YOLO detection file for a given image.

        Args:
            image_file (str): The filename of the image.

        Returns:
            str: The contents of the YOLO detection file as a string.
        """
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(self.labels_folder, label_file)

        if not os.path.isfile(label_path):
            return None

        with open(label_path, "r") as file:
            return file.read()

    def extract_roi(self, image, yolo_data):
        """
        Extract region of interest from an image based on YOLO data.

        Args:
            image (numpy.ndarray): The image to process.
            yolo_data (str): The YOLO detection data corresponding to the image.

        Returns:
            list: A list of region of interest images as numpy arrays.
        """
        class_index = self.get_class_index(self.class_name)
        if class_index is None or yolo_data is None:
            return []

        lines = yolo_data.strip().split('\n')
        roi_images = []

        for line in lines:
            data = line.strip().split()
            detected_class_index = int(data[0])
            if detected_class_index == class_index:
                x_center, y_center, box_width, box_height = map(float, data[1:])
                x = int((x_center - (box_width / 2)) * image.shape[1])
                y = int((y_center - (box_height / 2)) * image.shape[0])
                w = int(box_width * image.shape[1])
                h = int(box_height * image.shape[0])
                roi = image[y:y + h, x:x + w]
                roi_images.append(roi)

        return roi_images

    def save_roi(self, image_file, rois):
        """
        Save the extracted region of interest as separate images.

        Args:
        image_file (str): The filename of the original image.
        plastic_wrappers (list): A list of region of interest images as numpy arrays.
        """
        image_name = os.path.splitext(image_file)[0]
        for i, roi in enumerate(rois):
          output_file = f"{image_name}_{i}.jpg"
          output_path = os.path.join(self.output_folder, output_file)
          print(f"Saving image {i + 1} of {len(rois)} to {output_path}")
          save_image(roi, output_path)



    def get_class_index(self, class_name):
        """
        Get the index of the class from the classes.txt file.

        Args:
            class_name (str): The class name for which to extract region of interest.

        Returns:
            int: The index of the class or None if not found.
        """
        if not os.path.isfile(self.classes_file):
            return None

        with open(self.classes_file, "r") as file:
            classes = file.read().strip().split('\n')

        if class_name in classes:
            return classes.index(class_name)
        return None

    def process_images(self):
        """
        Process each image in the images folder, extract region of interest, and save them as separate images.
        """
        self.create_output_folder()

        # Process each image in the images folder
        for image_file in os.listdir(self.images_folder):
            image = self.load_image(image_file)
            yolo_data = self.read_yolo_detection_file(image_file)
            rois = self.extract_roi(image, yolo_data)
            self.save_roi(image_file, rois)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract region of interest from YOLO training dataset images.")
    parser.add_argument("--images_folder", type=str, help="Path to the folder containing input images.")
    parser.add_argument("--labels_folder", type=str, help="Path to the folder containing YOLO detection files.")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where the extracted plastic wrappers will be saved.")
    parser.add_argument("--class_name", type=str, help="The class name for which to extract plastic wrapper objects.")
    parser.add_argument("--classes_file", type=str, help="Path to the file containing class names.")
    args = parser.parse_args()

    # Create an instance of the RegionOfInterestExtractor class and process the images
    extractor = RegionOfInterestExtractor(args.images_folder, args.labels_folder, args.output_folder, args.class_name, args.classes_file)
    extractor.process_images()
