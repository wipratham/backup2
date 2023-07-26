import torch
import timm
from torchvision import transforms
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter

def load_model(model_name):
    """
    Load the pre-trained Vision Transformer model and freeze all layer weights.

    Args:
        model_name (str): Name of the pre-trained Vision Transformer model.

    Returns:
        torch.nn.Module: The pre-trained model with frozen weights.
    """
    model = timm.create_model(model_name, pretrained=True)
    model.head = torch.nn.Identity()

    # Freeze the weights for all layers
    for param in model.parameters():
        param.requires_grad = False

    return model

def extract_features(image_path, model):
    """
    Extract features from an image using the provided model.

    Args:
        image_path (str): The path to the input image file.
        model (torch.nn.Module): The pre-trained model.

    Returns:
        torch.Tensor: Extracted features as a tensor.
    """
    image = Image.open(image_path)
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    extracted_features = model(image.unsqueeze(0))
    return extracted_features

