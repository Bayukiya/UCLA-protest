import os
import torch

from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class LetterRecognitionDataset(Dataset):
    def __init__(self, datafile, is_train=True):
        self.data = pd.read_csv(datafile, header=None, sep=",")
        self.target = self.data.iloc[:, 0]
        self.is_train = is_train

        # 1. Split the raw dataframe first
        train_df = self.data.iloc[:16000]
        test_df = self.data.iloc[16000:]

        # 2. Extract features
        train_feat_raw = train_df.iloc[:, 1:17]
        test_feat_raw = test_df.iloc[:, 1:17]

        # 3. Calculate statistics ONLY from training features
        train_mean = train_feat_raw.mean()
        train_std = train_feat_raw.std()

        # 4. Apply training statistics to BOTH (This is the key!)
        train_feat_scaled = (train_feat_raw - train_mean) / train_std
        test_feat_scaled = (test_feat_raw - train_mean) / train_std

        # convert the letters (A-Z) into integers (0-25) labels: use LabelEncoder from sklearn or pandas categorical encoding
        all_labels = self.target.astype("category").cat.codes
        # 5. Assign BOTH features and labels based on the flag
        if self.is_train:
            self.features = torch.tensor(train_feat_scaled.values, dtype=torch.float32)
            self.target = torch.tensor(all_labels[:16000].values, dtype=torch.long)
        else:
            self.features = torch.tensor(test_feat_scaled.values, dtype=torch.float32)
            self.target = torch.tensor(all_labels[16000:].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # get the label and features for the given index
        label = self.target[idx]
        features = self.features[idx]
        return {
            "features": features,
            "label": label,
        }


if __name__ == "__main__":
    path = "/media/viplab/New Volume/Datasets/letter+recognition"
    train_dataset = LetterRecognitionDataset(os.path.join(path, "letter-recognition.data"))

    # create a DataLoader for the training dataset with a batch size of 64 and shuffling enabled
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    df = train_dataset.data

    print(f"Dataset size: {len(train_dataset)}")
    print(f"First sample: {train_dataset[0]}")

    print(f"Batch of features: {train_loader.dataset.features[:5]}")
    print(f"Batch of labels: {train_loader.dataset.target[:5]}")

    # Visualize the distribution of letters in the dataset
    print("Letter counts:")
    print(df.iloc[:, 0].value_counts().sort_index())

    print(df.iloc[:, 0].value_counts())
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.iloc[:, 1:].corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Data attributes overlap")
    plt.show()
