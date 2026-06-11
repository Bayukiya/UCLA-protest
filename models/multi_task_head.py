import torch.nn as nn


class MultiTaskProtestHead(nn.Module):
    def __init__(self, input_dim=2048, num_attributes=10):
        """
        Args:
            input_dim (int): The size of the incoming feature vector from your backbone.
                             (e.g., 2048 for ResNet50).
            num_attributes (int): The number of sparse multi-label traits (e.g., 10).
        """
        super(MultiTaskProtestHead, self).__init__()

        # Task 1: Binary Classification (Protest vs Non-Protest)
        # Outputs 2 values representing the raw scores (logits) for class 0 and class 1
        self.protest_head = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 2))

        # Task 2: Image Regression (Continuous Violence Score)
        # Outputs a single scalar value representing the violence score
        self.violence_head = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

        # Task 3: Multi-Label Attribute Prediction (10 distinct flags/masks/traits)
        # Outputs a 10-dimensional vector of raw logits (processed later via Sigmoid)
        self.attribute_head = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_attributes))

    def forward(self, features):
        """
        Processes the shared feature vector through each parallel stream.
        """
        protest_logits = self.protest_head(features)
        violence_score = self.violence_head(features)
        attribute_logits = self.attribute_head(features)

        # Return a dictionary matching the structure of your Dataloader batch keys
        return {"protest": protest_logits, "violence": violence_score, "attributes": attribute_logits}
