import torch.nn as nn
import torchvision.models as models
import timm


class MultiModelBackbone(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True):
        super(MultiModelBackbone, self).__init__()

        self.model_name = model_name

        # 1. Cleanly check if the user wants a TIMM model or a TORCHVISION model
        if "vit" in model_name or "efficientnet" in model_name or "resnest" in model_name:
            # Load straight from the timm library without using eval()
            # we set num_classes=0 to automatically strip away the classification head!
            self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.feature_extractor.num_features
        else:
            # Fallback to standard torchvision ResNets
            if model_name == "resnet50":
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                base_model = models.resnet50(weights=weights)
            elif model_name == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                base_model = models.resnet18(weights=weights)
            else:
                raise ValueError(f"Model {model_name} not explicitly configured.")

            self.feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.feature_extractor = base_model

    def forward(self, x):
        return self.feature_extractor(x)
