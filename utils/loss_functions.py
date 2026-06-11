import os
import pandas as pd

import torch


import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class JutePestDataset(Dataset):
    CLASSES = ["Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid", "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler", "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", "Scopula Emissaria", "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"]

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # convert those folder names into integers (0-8); a dictionary that maps your folder names to numbers.
        self.categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.categories)}

        for cls in self.categories:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}

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


base_transforms = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        *base_transforms,
    ]
)


val_test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        *base_transforms,
    ]
)


train_dataset = JutePestDataset(
    root_dir="/home/viplab/Documents/manuscript_draft/Jute_Pest_Dataset/train",
    transform=train_transforms,
)
print(train_dataset.image_paths[:5])
print(train_dataset.categories)
print(train_dataset.class_to_idx)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
data_iter = iter(train_loader)
batch = next(data_iter)
train_dataset.show_batch(batch)

print(f"Train images ready: {len(train_dataset)}")

val_dataset = JutePestDataset(
    root_dir="/home/viplab/Documents/manuscript_draft/Jute_Pest_Dataset/val",
    transform=val_test_transforms,
)
print(val_dataset.image_paths[:5])
print(val_dataset.categories)
print(val_dataset.class_to_idx)
print(len(val_dataset))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
data_iter = iter(val_loader)
batch = next(data_iter)
val_dataset.show_batch(batch)

assert train_dataset.class_to_idx == val_dataset.class_to_idx, "Error: Class mappings do not match!"
print("Success: Train and Val class mappings are identical.")


print(f"Validation images found: {len(val_dataset)}")

test_dataset = JutePestDataset(
    root_dir="/home/viplab/Documents/manuscript_draft/Jute_Pest_Dataset/test",
    transform=val_test_transforms,
)
print(test_dataset.image_paths[:5])
print(test_dataset.categories)
print(test_dataset.class_to_idx)
print(len(test_dataset))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
data_iter = iter(test_loader)
batch = next(data_iter)
test_dataset.show_batch(batch)


print(f"Test images ready: {len(test_dataset)}")

# Print the count of images in each folder of your train split
counts = pd.Series(train_dataset.labels).value_counts().sort_index()
print(f"--- Train class distribution ({len(train_dataset)} total) ---")
for idx, count in counts.items():
    class_name = train_dataset.categories[idx]
    print(f"{class_name:<30}: {count} images")


# 1. Get counts sorted by index to match category order
class_counts = pd.Series(train_dataset.labels).value_counts().sort_index()
counts = class_counts.values
class_names = train_dataset.categories

# 2. Create the bar chart
plt.figure(figsize=(14, 7))
bars = plt.bar(class_names, counts, color="teal", alpha=0.8)

# 3. Add styling to match paper-quality reporting
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.title("Class Distribution of Jute Pest Dataset (Training Set)", fontsize=14, pad=20)
plt.xlabel("Pest Species", fontsize=12)
plt.ylabel("Image Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()
