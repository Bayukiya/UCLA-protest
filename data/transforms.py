import os
import torch
import glob
import pandas as pd
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns


class RealWasteDataset(Dataset):
    def __init__(self, labels, root_dir, transform=None):
        """
        Args:
            labels (list): List of labels corresponding to the subfolders in the root directory.
            root_dir (string): Directory containing all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # the image names are in folders. root folder is 'RealWaste, under it thare 9 subfolders (0-8) and under each subfolder there are images.
        # there is no txt or csv file. So we need to create a dataframe from the folder structure. The folder name is the label and the image name is the file name.
        # 1. create a list of all image file paths and their corresponding labels based on the folder structure.
        image_paths = []
        labels = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    image_paths.append(img_path)
                    labels.append((label))
        # 2. create a dataframe from the list of image paths and labels.
        self.file_name = pd.DataFrame({"fname": image_paths, "label": labels})

        # convert those folder names into integers (0-8); a dictionary that maps your folder names to numbers.
        self.categories = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.categories)}

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        img_name = self.file_name.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")
        label = self.file_name.iloc[idx, 1]
        label = self.class_to_idx[label]  # convert the label to numeric using the class_to_idx mapping

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),  # wrap them in torch.tensor()
        }

    def show_batch(self, sample_batch):
        images_batch = sample_batch["image"]
        labels_batch = sample_batch["label"]
        batch_size = len(images_batch)

        # create a grid for the 4 images in your batch and display them with their labels
        fig, axes = plt.subplots(1, batch_size, figsize=(12, 6))
        for i in range(batch_size):
            plt.subplot(1, batch_size, i + 1)
            # 2. Rearrange dimensions: [C, H, W] -> [H, W, C]
            img = images_batch[i].permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # unnormalize
            img = img.clip(0, 1)  # clip to valid range
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {labels_batch[i].item()}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    def show_average_image(self, img_dir):
        images = []
        # 1. Grab all image files
        files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg"))]

        for f in files[:100]:  # Look at first 100 for speed
            img = Image.open(os.path.join(img_dir, f)).convert("RGB").resize((224, 224))
            images.append(np.array(img))

        # 2. Calculate the "Mean" across the whole stack
        avg_img = np.mean(images, axis=0).astype(np.uint8)

        # 3. Visualize the "Essence" of the class
        plt.imshow(avg_img)
        plt.title(f"The 'Ghost' of {os.path.basename(img_dir)}")
        plt.axis("off")
        plt.show()

    def get_average_image(self, img_dir, target_size=(224, 224), n_samples=50):
        images = []
        # Support common image extensions
        valid_exts = (".jpg", ".jpeg", ".png")
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]

        # Use a sample for speed (50 images is usually enough to see the 'ghost')
        sample_files = files[:n_samples]

        for f in sample_files:
            try:
                img = Image.open(os.path.join(img_dir, f)).convert("RGB").resize(target_size)
                images.append(np.array(img))
            except Exception:
                continue

        if not images:
            return None

        # Return the mathematical mean across the image stack
        return np.mean(images, axis=0).astype(np.uint8)

    def compare_ghosts(self, dir_a, dir_b, name_a, name_b):
        avg_a = self.get_average_image(dir_a)
        avg_b = self.get_average_image(dir_b)

        if avg_a is None or avg_b is None:
            print("Error: Check your folder paths!")
            return

        # Plotting the 'Ghosts' side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(avg_a)
        axes[0].set_title(f"The 'Ghost' of {name_a}")
        axes[0].axis("off")

        axes[1].imshow(avg_b)
        axes[1].set_title(f"The 'Ghost' of {name_b}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


train_path = "/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste"
my_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
)
# crete an instance of the dataset
train_dataset = RealWasteDataset(labels=list(range(9)), root_dir=train_path, transform=my_transform)
print(f"train_dataset.__len__(): {train_dataset.file_name}")
print(f"train_dataset.__getitem__(1): {train_dataset.root_dir}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
print(f"train_loader.__len__(): {len(train_loader)} batches of size 4   each    batch has 4 samples")

for batch in train_loader:
    print("image", batch["image"].shape)
    print("label", batch["label"].shape)
    break

# Visualize a batch of data
data_iter = iter(train_loader)
batch = next(data_iter)
train_dataset.show_batch(batch)

counts = train_dataset.file_name["label"].value_counts()
print("Label distribution in the dataset:")
print(counts)
counts.plot(kind="bar", color="skyblue")
plt.title("My Local RealWaste Balance")
plt.ylabel("Number of Images")
plt.xlabel("Label")
plt.show()


# train_dataset.show_average_image("/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste/Textile Trash")

# --- RUN THE TEST ---
base_path = "/home/viplab/Documents/manuscript_draft/realwaste-main/RealWaste"

# Compare Glass and Plastic to see their visual overlap
train_dataset.compare_ghosts(os.path.join(base_path, "Textile Trash"), os.path.join(base_path, "Paper"), "Textile Trash", "Paper")
