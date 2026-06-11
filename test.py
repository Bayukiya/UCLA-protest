import torch
from torch.utils.data import DataLoader
import os

# Import your custom modules
from data.dataset import UCLADataset
from data.transforms import get_protest_transforms
from models.backbone import MultiModelBackbone
from models.multi_task_head import MultiTaskProtestHead


def run_integration_test():
    print("🚀 Initializing Integration Test Pipeline...")

    # 1. Setup paths and mock transforms
    train_path = "/media/viplab/New Volume/Datasets/UCLA-protest/img/"
    test_transforms = get_protest_transforms(is_train=False, image_size=224)

    # 2. Instantiate Dataset and Dataloader
    print("📦 Loading UCLA-Protest Dataset instance...")
    dataset = UCLADataset(
        file_name="/media/viplab/New Volume/Datasets/UCLA-protest/annot_train.txt",
        img_dir=os.path.join(train_path, "train"),
        transform=test_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # Extract a single mock batch from dataloader
    data_iter = iter(dataloader)
    batch = next(data_iter)

    images = batch["image"]
    print(f"✅ Input batch captured. Image Tensor shape: {images.shape}")

    # 3. Instantiate the Model Components
    print("🏗️ Initializing ResNet50 Backbone and Multi-Task Heads...")
    backbone = MultiModelBackbone(model_name="resnet50", pretrained=False)  # Speed up test by skipping downloading weights
    head = MultiTaskProtestHead(input_dim=backbone.feature_dim, num_attributes=10)

    # Set models to evaluation mode (turns off dropout)
    backbone.eval()
    head.eval()

    # 4. Perform the Forward Pass Forward Thread
    print("⚡ Running forward pass through network blocks...")
    with torch.no_grad():  # Disable gradient calculations for safety
        features = backbone(images)
        predictions = head(features)

    print("\n🎉 FORWARD PASS SUCCESSFUL! Printing predictions metadata:")
    print(f"➡️ Extract feature shape from backbone: {features.shape}")
    print(f"➡️ Task 1 (Protest Logits) Shape:       {predictions['protest'].shape}")
    print(f"➡️ Task 2 (Violence Score) Shape:      {predictions['violence'].shape}")
    print(f"➡️ Task 3 (Attribute Logits) Shape:     {predictions['attributes'].shape}")
    print("\n💪 Your full model integration is completely seamless!")


if __name__ == "__main__":
    run_integration_test()
