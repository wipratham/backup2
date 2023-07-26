import os
import argparse

class ClusterLabelUpdater:
    def __init__(self, clusters_folder, labels_folder, class_name, classes_file, new_classes_file, clusters):
        self.clusters_folder = clusters_folder
        self.labels_folder = labels_folder
        self.clusters_dict = {}
        self.class_name = class_name
        self.classes_file = classes_file
        self.new_classes_file = new_classes_file
        self.clusters = clusters
        self.load_clusters()
        self.skip_classes_below = self.find_class_line_number()
        self.class_indices = self.find_class_indices()

    def load_clusters(self):
        for cluster_folder in os.listdir(self.clusters_folder):
            cluster_folder_path = os.path.join(self.clusters_folder, cluster_folder)
            if not os.path.isdir(cluster_folder_path):
                continue
            
            image_files = os.listdir(cluster_folder_path)
            self.clusters_dict[cluster_folder] = image_files

    def find_class_line_number(self):
        with open(self.classes_file, "r") as file:
            classes = file.read().splitlines()
            try:
                line_number = classes.index(self.class_name)
            except ValueError:
                line_number = -1  # Class not found
        return line_number

    def find_class_indices(self):
        with open(self.new_classes_file, "r") as file:
            classes = file.read().splitlines()

        class_indices = {}
        for cluster in self.clusters:
            if cluster in classes:
                class_indices[cluster] = classes.index(cluster)

        return class_indices

    def update_labels(self):
        for label_file in os.listdir(self.labels_folder):
            if not label_file.endswith(".txt"):
                continue

            is_frame_file = label_file.startswith("frame_")
            is_image_file = label_file.startswith("image_")

            if not is_frame_file and not is_image_file:
                continue

            label_path = os.path.join(self.labels_folder, label_file)

            with open(label_path, "r+") as file:
                lines = file.readlines()
                file.seek(0)  # Reset file pointer to the beginning
                file.truncate()  # Clear the file contents

                for line in lines:
                    class_index, x_center, y_center, box_width, box_height = map(float, line.strip().split())

                    # Skip updating the class label if it is below the specified threshold
                    if int(class_index) < self.skip_classes_below:
                        file.write(line)  # Keep the line as it is
                        continue

                    # Update the class index based on the assigned cluster
                    image_number = label_file.split("_")[1].split(".")[0]
                    if is_frame_file:
                        image_key = f"frame_{image_number}.jpg"
                    elif is_image_file:
                        image_key = f"image_{image_number}.jpg"

                    # Check which cluster the image belongs to
                    cluster_found = False
                    for cluster, image_list in self.clusters_dict.items():
                        if image_key in image_list and cluster in self.class_indices:
                            cluster_found = True
                            new_class_index = self.class_indices[cluster]
                            break

                    if not cluster_found:
                        # If the image is not found in any cluster, keep the class index as it is
                        new_class_index = int(class_index)

                    line = line.replace(str(class_index), str(new_class_index))

                    # Write the updated line to the label file
                    file.write(f"{int(new_class_index)} {x_center} {y_center} {box_width} {box_height}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters-folder', dest='clusters_folder', metavar='', help='Path to the folder containing cluster subfolders')
    parser.add_argument('--labels-folder', dest='labels_folder', metavar='', help='Path to the folder containing YOLO label files')
    parser.add_argument('--class-name', dest='class_name', metavar='', help='Name of the class to skip classes below')
    parser.add_argument('--classes-file', dest='classes_file', metavar='', help='Path to the classes file')
    parser.add_argument('--new-classes-file', dest='new_classes_file', metavar='', help='Path to the new classes file')
    parser.add_argument('--clusters', dest='clusters', nargs='+', metavar='', help='Names of the clusters')
    args = parser.parse_args()

    if args.clusters_folder and args.labels_folder and args.class_name and args.classes_file and args.new_classes_file and args.clusters:
        cluster_label_updater = ClusterLabelUpdater(args.clusters_folder, args.labels_folder, args.class_name, args.classes_file, args.new_classes_file, args.clusters)
        cluster_label_updater.update_labels()
        print("Labels updated successfully.")
    else:
        print("Invalid arguments provided. Please provide --clusters-folder, --labels-folder, --class-name, --classes-file, --new-classes-file, and --clusters.")

if __name__ == '__main__':
    main()
