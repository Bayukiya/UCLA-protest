import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, attribute_pos_weights=None):
        """
        Args:
            attribute_pos_weights (Tensor): A 10-dimensional tensor containing the penalty
                                            multiplier for each sparse attribute.
        """
        super(MultiTaskLoss, self).__init__()

        # 1. Define the core sub-loss equations
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.MSELoss()

        # 2. Add positive weighting to the multi-label loss to handle severe class imbalance
        if attribute_pos_weights is not None:
            self.attribute_loss_fn = nn.BCEWithLogitsLoss(pos_weight=attribute_pos_weights)
        else:
            self.attribute_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (dict): Output dictionary from MultiTaskProtestHead
            targets (dict): Raw target dictionary from your Dataloader batch
        """
        # Calculate individual task errors
        loss_protest = self.classification_loss_fn(predictions["protest"], targets["label"])
        loss_violence = self.regression_loss_fn(predictions["violence"].squeeze(-1), targets["violance_score"])
        loss_attributes = self.attribute_loss_fn(predictions["attributes"], targets["attributes"])

        # 3. Combine losses using a balanced linear summation strategy
        # You can adjust these multipliers (weights) if one task dominates training
        total_loss = (1.0 * loss_protest) + (1.0 * loss_violence) + (1.0 * loss_attributes)

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_protest": loss_protest.item(),
            "loss_violence": loss_violence.item(),
            "loss_attributes": loss_attributes.item(),
        }
