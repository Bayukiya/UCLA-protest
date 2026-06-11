import os
import torch
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, transform=None, is_train=True):
        self.root_dir = os.path.join(root_dir, category)
        self.transform = transform
        self.is_train = is_train

        self.image_paths = []
        self.labels = []
        self.ground_truth_paths = []

        if is_train:
            # For training, we only want to load images from the 'train' subfolder
            train_dir = os.path.join(self.root_dir, "train", "good")  # all training images are in the 'good' subfolder
            self.image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
            self.labels = [0] * len(self.image_paths)  # all training images are labeled as 0 (normal)
        else:
            # For testing, we want to load images from the 'test' subfolder and their corresponding ground truth masks
            test_dir = os.path.join(self.root_dir, "test")
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    print(f"Skipping non-directory: {defect_dir}")
                    continue
                print(f"Processing directory: {defect_dir}")

                for img_name in os.listdir(defect_dir):
                    self.image_paths.append(os.path.join(defect_dir, img_name))
                    if defect_type == "good":
                        self.labels.append(0)  # normal images are labeled as 0
                    else:
                        self.labels.append(1)  # defective images are labeled as 1
                    # localization: Handle ground truth masks for defective images
                    if defect_type == "good":
                        self.ground_truth_paths.append(None)  # no mask for normal images
                    else:
                        mask_name = img_name.split(".")[0] + "_mask.png"
                        mask_path = os.path.join(self.root_dir, "ground_truth", defect_type, mask_name)
                        if os.path.exists(mask_path):
                            self.ground_truth_paths.append(mask_path)
                        else:
                            print(f"Warning: Ground truth mask not found for {img_name} at {mask_path}")
                            self.ground_truth_paths.append(None)  # handle missing masks gracefully

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB")

        label = self.labels[idx]
        mask_path = self.ground_truth_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")  # load mask as grayscale
        else:
            mask = Image.new("L", image.size)  # create an empty mask if none exists

        if self.transform:
            image = self.transform(image)
            mask_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
            mask = mask_transform(mask)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "ground_truth": mask,
        }

    def show_batch(self, sample_batch):
        images_batch = sample_batch["image"]
        labels_batch = sample_batch["label"]
        masks_batch = sample_batch["ground_truth"]  # This was 'mask' before
        batch_size = len(images_batch)

        # Create 2 rows: Top for images, Bottom for masks
        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))

        for i in range(batch_size):
            # --- 1. Plot Original Image ---
            img = images_batch[i].permute(1, 2, 0).numpy()
            # Unnormalize (using ImageNet stats)
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img = img.clip(0, 1)

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Label: {labels_batch[i].item()} (Normal)" if labels_batch[i].item() == 0 else f"Label: {labels_batch[i].item()} (Anomaly)")
            axes[0, i].axis("off")

            # --- 2. Plot Ground Truth Mask ---
            mask = masks_batch[i].squeeze().numpy()  # [1, H, W] -> [H, W]
            axes[1, i].imshow(mask, cmap="gray")
            axes[1, i].set_title("GT Mask")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()


path = "/media/viplab/New Volume/Datasets/mvtec_ad"
category = "toothbrush"
base_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_dataset = MVTecDataset(root_dir=path, category=category, transform=base_transform, is_train=False)
print(train_dataset.image_paths[:5])
print(train_dataset.labels[:5])
print(train_dataset.ground_truth_paths[:5])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
data_iter = iter(train_loader)
batch = next(data_iter)
train_dataset.show_batch(batch)
