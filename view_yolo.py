import os
import cv2

# Path to the folder containing images
images_folder = "/home/wi/Sheel/yolo_final/images"

# Path to the folder containing YOLO annotation files
annotations_folder = "/home/wi/Sheel/yolo_final/label_trial"

# Path to the classes file
classes_file = "classes.txt"

# Output folder to save the annotated images
output_folder = "output_folder"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the classes from the file
with open(classes_file, "r") as file:
    classes = file.read().splitlines()

# Process each annotation file and find the corresponding image
for annotation_file in os.listdir(annotations_folder):
    if not annotation_file.endswith(".txt"):
        continue

    annotation_path = os.path.join(annotations_folder, annotation_file)

    # Determine the corresponding image file name
    image_file = annotation_file.replace(".txt", ".png")
    image_path = os.path.join(images_folder, image_file)

    # Check if the corresponding image file exists
    if not os.path.exists(image_path):
        continue

    # Load the image
    image = cv2.imread(image_path)

    # Read the YOLO annotations from the file
    with open(annotation_path, "r") as file:
        annotations = file.read().splitlines()

    # Draw the annotations on the image
    for annotation in annotations:
        class_index, x_center, y_center, width, height = map(float, annotation.split())

        # Convert the coordinates to pixel values
        img_height, img_width, _ = image.shape
        x_center = int(x_center * img_width)
        y_center = int(y_center * img_height)
        width = int(width * img_width)
        height = int(height * img_height)

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Get the class label
        class_label = classes[int(class_index)]

        # Draw the bounding box rectangle and class label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image in the output folder
    output_image_path = os.path.join(output_folder, f"annotated_{image_file}")
    cv2.imwrite(output_image_path, image)

print("Annotated images saved in the output folder.")
