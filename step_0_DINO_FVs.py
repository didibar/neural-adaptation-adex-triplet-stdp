import torch
from torch.hub import load
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import copy

# Build a normalized image feature matrix using a pre-trained DINO model.
def prepare_input_matrix(image_folder, directory):

    # Load the pre-trained DINO model with a Vision Transformer backbone.
    dino = load('facebookresearch/dino:main', 'dino_vitb16')

    # Define the preprocessing pipeline expected by the DINO model.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                                   # Resize each image to 224x224 pixels.
        transforms.ToTensor(),                                           # Convert the image to a tensor.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values.
    ])

    # Extract the feature vector from a single image using the DINO model.
    def extract_features(image_path, model):
        image = Image.open(image_path).convert("RGB")  # Open the image and convert it to RGB.
        image = transform(image).unsqueeze(0)          # Apply preprocessing and add the batch dimension.

        with torch.no_grad():        # Disable gradient computation during inference.
            features = model(image)  # Compute the image feature vector.

        return features.squeeze().numpy()  # Return the feature vector as a NumPy array.

    # Collect all JPG images from the input folder.
    image_files = list(image_folder.glob("*.jpg"))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No .jpg images found in: {directory}")

    # Extract features for all images in the folder.
    features_list = []  # Store the extracted feature vectors.
    image_paths = []    # Store the corresponding image file names.

    for img_file in image_files:
        features = extract_features(img_file, dino)
        features_list.append(features)
        image_paths.append(img_file.name)

    # Convert the extracted features and image names to NumPy arrays.
    features = np.array(features_list)
    image_paths = np.array(image_paths)

    # Create a DataFrame with image names and their corresponding feature vectors.
    df = pd.DataFrame({
        "image_name": image_paths,
        "features": list(features)
    })

    # Define the expected subcategories and their sorting order.
    subcategory_order = ["wild_animals", "fruit", "flowers", "insects", "birds",
                        "manmade_food", "clothes", "furniture", "instruments", "computer"]

    # Assign each image to a subcategory based on its filename.
    def assign_subcategory(image_name):
        for subcategory in subcategory_order:
            if subcategory != "unknown" and subcategory in image_name:
                return subcategory
        return "unknown"

    # Add subcategory labels and sort the DataFrame accordingly.
    df["subcategory"] = df["image_name"].apply(assign_subcategory)
    df["subcategory"] = pd.Categorical(df["subcategory"], categories=subcategory_order, ordered=True)
    df = df.sort_values(["subcategory", "image_name"]).reset_index(drop=True)

    # Create a copy of the extracted features before shifting their values.
    df["features_upper"] = df["features"].apply(lambda x: copy.deepcopy(x))

    # Find the global minimum feature value to shift all vectors to non-negative values.
    minimum = 0
    for i in range(len(df)):
        ap = min(df["features"][i])
        if ap < minimum:
            minimum = ap

    # Shift all feature values using the global minimum.
    for i in range(len(df)):
        for j in range(len(df["features_upper"][i])):
            df["features_upper"][i][j] += -minimum


    # Create a copy of the shifted features for min-max normalization.
    df["features_normalized"] = df["features_upper"].apply(lambda x: copy.deepcopy(x))

    # Find the global maximum value after shifting.
    maximum = 0
    for i in range(len(df)):
        ap = max(df["features_upper"][i])
        if ap > maximum:
            maximum = ap

    # Normalize all feature values to the range [0, 1].
    for i in range(len(df)):
        for j in range(len(df["features_normalized"][i])):
            df["features_normalized"][i][j] = df["features_normalized"][i][j] / maximum


    # Stack normalized feature vectors into a matrix and save it as a NumPy file.
    feature_matrix = {'features_normalized': np.vstack(df["features_normalized"].values)}
    np.save(directory / "X_features_normalized.npy", feature_matrix["features_normalized"])

    return feature_matrix["features_normalized"]
