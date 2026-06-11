import os
import torch
import glob
import pandas as pd
import PIL.Image as Image
from torchvision import datasets, transforms


def show_directory_structure(root_dir):
    # Walk through the directory structure and print out the names of all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Directory: {dirpath}")
        for dirname in dirnames:
            print(f"  Subdirectory: {dirname}")
        for filename in filenames:
            print(f"  File: {filename}")


def use_glob_to_list_files(root_dir):
    # Use glob to list all files in the directory and its subdirectories
    file_paths = glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
    # for file_path in file_paths:
    #     print(f"File: {file_path}")
    # extract the label from the string path (using os.path.basename or string splitting)
    labels = [os.path.basename(os.path.dirname(file_path)) for file_path in file_paths]
    print(f"Labels: {pd.Series(labels).unique()}")


# production-ready datasets.ImageFolder usage
def test_image_folder(root_dir):
    # Define the transformations to be applied to the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create an ImageFolder dataset
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    # Print out the class names and the number of samples in each class
    print(f"Classes: {dataset.classes}")
    # print the number of samples in each class in numerically ascending order based on index of the class (which is based on alphabetical order of the class names)
    class_counts = pd.Series(dataset.targets).value_counts()
    print(f"Class counts: {class_counts}")
    print("dataset", dataset.class_to_idx)
    # print tuble of (image, label) for the first 5 samples in the dataset
    for i in range(5):
        image, label = dataset[i]
        print(f"Sample {i}: Image shape: {image.shape}, Label: {label}")


# show_directory_structure("/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste")
use_glob_to_list_files("/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste")

# test_image_folder("/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste")
test_image_folder("/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste")
