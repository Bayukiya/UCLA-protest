import os
import torch
import pandas as pd
import PIL.Image as Image
import seaborn as sns

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class UCLADataset(Dataset):
    def __init__(self, file_name, img_dir, transform=None):
        """ "
        args:
            file_name (string): path to the txt file containing image names, labels and other information
            img_dir (string): path to the directory containing all the images
            transform (callable, optional): optional transform to be applied on a sample
        """
        # 1. loaad (read) the txt file using pandas and tab as the separator
        self.file_name = pd.read_csv(file_name, sep="\t")
        # 2. clean the data frame by replacing '-' with 0 and convert to float; so model can handle it.
        self.file_name = self.file_name.replace("-", 0)
        # 3. ensure all columns except 'fname' are numeric.
        cols = self.file_name.drop("fname", axis=1).columns
        self.file_name[cols] = self.file_name[cols].apply(pd.to_numeric)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_name.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.file_name.iloc[idx, 1]
        violance_score = self.file_name.iloc[idx, 2]
        attributes = self.file_name.iloc[idx, 3:13].values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "violance_score": torch.tensor(violance_score, dtype=torch.float),
            "violence_mask": torch.tensor(label, dtype=torch.float),
            "attributes": torch.tensor(attributes, dtype=torch.float),
        }

    def show_batch(sample_batch):
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


train_path = "/media/viplab/New Volume/Datasets/UCLA-protest/img/"
my_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
)
# crete an instance of the dataset
train_dataset = UCLADataset(
    file_name="/media/viplab/New Volume/Datasets/UCLA-protest/annot_train.txt",
    img_dir=os.path.join(train_path, "train"),
    transform=my_transform,
)
# create a dataloader for the dataset
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# Visualize a batch of data
data_iter = iter(train_loader)
batch = next(data_iter)
UCLADataset.show_batch(batch)

# select only the label column
label_cols = train_dataset.file_name.drop(columns=["fname"], errors="ignore")
# Calculate the correlation matrix
correlation_matrix = label_cols.corr()
# Plot the correlation matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Between Protest Attributes")
plt.show()

# Identifying the Sparse classes:
# drop the filename and non-binary columns for correlation analysis
attributes = train_dataset.file_name.drop(columns="fname", errors="ignore")
# sum the columns to get the count of '1's for each attribute
attribute_counts = attributes.sum().sort_values(ascending=False)
# plot the counts using seaborn barplot
plt.figure(figsize=(12, 6))
sns.barplot(x=attribute_counts.index, y=attribute_counts.values, palette="viridis", hue=0.5)
plt.xticks(rotation=45)
plt.title("Frequency of Attributes in UCLA-Protest Dataset (Training Set)")
plt.xlabel("Number of Occurences (Images with Attribute=1)")
plt.ylabel("Attribute")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()

for batch in train_loader:
    print("image", batch["image"].shape)
    print("label", batch["label"].shape)
    print("violance_score", batch["violance_score"].shape)
    print("violence_mask", batch["violence_mask"].shape)
    print("attributes", batch["attributes"].shape)
    break
