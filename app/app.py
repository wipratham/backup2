import argparse
import os
import sys

def run_feature_extraction(model_name, folder_path, log_dir):
    if model_name == "resnet50":
        from feature_extraction.resnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "resnet101":
        from feature_extraction.resnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "resnet110":
        from feature_extraction.resnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "resnet152":
        from feature_extraction.resnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "resnet164":
        from feature_extraction.resnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "resnet1202":
        from feature_extraction.resnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "vgg16":
        from feature_extraction.vgg_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "vgg19":
        from feature_extraction.vgg_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "vit_base_patch16_224":
        from feature_extraction.vit_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    elif model_name == "efficientnet_b0":
        from feature_extraction.efficientnet_feature_extraction import FeatureExtractor as ModelFeatureExtractor
    else:
        raise ValueError("Invalid model_name. Supported models: resnet, vgg, vit, efficientnet")

    print(f"Model Name: {model_name}")  # Add this line to check the imported model name

    # Create an instance of the FeatureExtractor class and process the images
    feature_extractor = ModelFeatureExtractor(model_name, folder_path, log_dir)
    feature_extractor.process_images()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run feature extraction using a pre-trained model.")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet50", "resnet101","resnet110","resnet152","resnet164","resnet1202","vgg16", "vgg19", "vit_base_patch16_224", "efficientnet_b0"],
                        help="Name of the pre-trained model (resnet, vgg, vit, efficientnet).")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the directory for storing TensorBoard logs.")
    args = parser.parse_args()

    # Add the parent directory to the system path to import modules correctly
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    # Run feature extraction based on the selected model
    run_feature_extraction(args.model_name, args.folder_path, args.log_dir)
