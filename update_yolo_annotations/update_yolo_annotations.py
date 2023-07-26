import os
import argparse

class ClusterLabelUpdater:
    def __init__(self, clusters_folder, labels_folder, class_name, classes_file, new_classes_file, cluster1, cluster2):
        self.clusters_folder = clusters_folder
        self.labels_folder = labels_folder
        self.clusters_dict = {}
        self.class_name = class_name
        self.classes_file = classes_file
        self.new_classes_file = new_classes_file
        self.cluster1 = cluster1
        self.cluster2 = cluster2
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
        if self.cluster1 in classes:
            class_indices[self.cluster1] = classes.index(self.cluster1)
        if self.cluster2 in classes:
            class_indices[self.cluster2] = classes.index(self.cluster2)

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

                class1_count = 0

                for line in lines:
                    class_index, x_center, y_center, box_width, box_height = map(float, line.strip().split())

                    # Skip updating the class label if it is below the specified threshold
                    if int(class_index) < self.skip_classes_below:
                        file.write(line)  # Keep the line as it is
                        continue

                    # Update the class index based on the assigned cluster
                    if int(class_index) == self.skip_classes_below:
                        image_number = label_file.split("_")[1].split(".")[0]
                        if is_frame_file:
                            image_key = f"frame_{image_number}_{class1_count}.jpg"
                        elif is_image_file:
                            image_key = f"image_{image_number}_{class1_count}.jpg"

                        # Check which cluster the image belongs to
                        cluster_found = False
                        for cluster, image_list in self.clusters_dict.items():
                            if image_key in image_list:
                                cluster_found = True
                                new_class_index = int(cluster[-1])
                                break

                        if not cluster_found:
                            # If the image is not found in any cluster, keep the class index as it is
                            new_class_index = int(class_index)

                        line = line.replace(str(class_index), str(new_class_index))
                        class1_count += 1

                    else:
                        # Update other class indices by adding 1
                        new_class_index = int(class_index) + 1
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
    parser.add_argument('--cluster1', dest='cluster1', metavar='', help='Class name in cluster 1')
    parser.add_argument('--cluster2', dest='cluster2', metavar='', help='Class name in cluster 2')
    args = parser.parse_args()

    if args.clusters_folder and args.labels_folder and args.class_name and args.classes_file and args.new_classes_file and args.cluster1 and args.cluster2:
        cluster_label_updater = ClusterLabelUpdater(args.clusters_folder, args.labels_folder, args.class_name, args.classes_file, args.new_classes_file, args.cluster1, args.cluster2)
        cluster_label_updater.update_labels()
        print("Labels updated successfully.")
    else:
        print("Invalid arguments provided. Please provide --clusters-folder, --labels-folder, --class-name, --classes-file, --new-classes-file, --cluster1, and --cluster2.")

if __name__ == '__main__':
    main()
