import os
import glob
import torch
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class DefungiDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        image_paths = glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
        labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]
        print(f"Unique labels found: {pd.Series(labels).unique()}")

        self.file_name = pd.DataFrame({"fname": image_paths, "label": labels})

        # convert those folder names into integers (0-8); a dictionary that maps your folder names to numbers.
        self.categories = sorted(pd.Series(labels).unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.categories)}

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        img_name = self.file_name.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")

        label = self.file_name.iloc[idx, 1]
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
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


train_dataset = DefungiDataset(
    root_dir="/home/viplab/Documents/manuscript_draft/defungi",
    transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)
print(train_dataset.file_name.head())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
data_iter = iter(train_loader)
batch = next(data_iter)
train_dataset.show_batch(batch)

counts = train_dataset.file_name["label"].value_counts()
print("Label distribution in the dataset:")
print(counts)
counts.plot(kind="bar", color="skyblue")
plt.title("My Local Defungi Balance")
plt.ylabel("Number of Images")
plt.xlabel("Label")
plt.show()
