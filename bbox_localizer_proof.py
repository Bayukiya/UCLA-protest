# bbox_localizer_proof.py

from __future__ import annotations

import os
import argparse
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from data.transforms import get_protest_transforms
from models.backbone import MultiModelBackbone
from models.multi_task_head import CrossAttentionMultiTaskHead
from models.model_factory import build_experiment_model
from utils.output_adapter import standardize_outputs


ATTRIBUTE_NAMES = [
    "Sign",
    "Photo",
    "Fire",
    "Police",
    "Children",
    "Group_20",
    "Group_100",
    "Flag",
    "Night",
    "Shouting",
]


SUPPORTED_BACKBONES = [
    "resnet50",
    "efficientnet_b3",
    "vit_base",
    "convnext_base",
]


def infer_experiment_mode_from_checkpoint_path(checkpoint_path: str) -> Optional[str]:
    """
    Infer unified experiment_mode from checkpoint parent directory.

    Example:
        checkpoints/unified/modern_resnet50_attention_unified_bce_balanced_multitask/best_model.pt

    returns:
        modern_resnet50_attention
    """

    parent_name = os.path.basename(os.path.dirname(checkpoint_path))

    known_modes = [
        "paper_released_resnet50",
        "modern_resnet50_parallel",
        "modern_resnet50_attention",
        "modern_efficientnet_b3_parallel",
        "modern_efficientnet_b3_attention",
        "modern_vit_base_parallel",
        "modern_vit_base_attention",
        "modern_convnext_base_parallel",
        "modern_convnext_base_attention",
    ]

    for mode in known_modes:
        if parent_name.startswith(mode):
            return mode

    return None


def infer_model_choice_from_experiment_mode(experiment_mode: str) -> str:
    """
    Convert unified experiment_mode to backbone name.
    """

    if "efficientnet_b3" in experiment_mode:
        return "efficientnet_b3"

    if "vit_base" in experiment_mode:
        return "vit_base"

    if "convnext_base" in experiment_mode:
        return "convnext_base"

    if "resnet50" in experiment_mode:
        return "resnet50"

    raise ValueError(f"Cannot infer backbone from experiment_mode='{experiment_mode}'.")


def infer_experiment_mode_and_model_choice(
    checkpoint_path: str,
    checkpoint: Dict,
) -> Tuple[str, str]:
    """
    Robustly infer experiment_mode and model_choice from:
        1. unified checkpoint metadata
        2. checkpoint parent directory
        3. legacy model_choice field
        4. filename fallback
    """

    experiment_mode = checkpoint.get("experiment_mode") or checkpoint.get("model_type") or checkpoint.get("run_mode")

    if experiment_mode is None:
        experiment_mode = infer_experiment_mode_from_checkpoint_path(checkpoint_path)

    if experiment_mode is not None:
        model_choice = infer_model_choice_from_experiment_mode(experiment_mode)
        return experiment_mode, model_choice

    model_choice = checkpoint.get("model_choice", None)

    if model_choice is not None:
        experiment_mode = f"modern_{model_choice}_attention"
        return experiment_mode, model_choice

    filename = os.path.basename(checkpoint_path)
    parent_name = os.path.basename(os.path.dirname(checkpoint_path))
    searchable_name = f"{parent_name}/{filename}"

    for backbone_name in SUPPORTED_BACKBONES:
        if backbone_name in searchable_name:
            experiment_mode = f"modern_{backbone_name}_attention"
            return experiment_mode, backbone_name

    raise ValueError(f"Could not determine model type from checkpoint metadata, parent folder, or filename. checkpoint_path={checkpoint_path}")


def build_attention_model_from_checkpoint(
    checkpoint_path: str,
    checkpoint: Dict,
    device: torch.device,
):
    """
    Build and load an attention model from either:
        - unified checkpoint: checkpoint['model_state']
        - legacy checkpoint: checkpoint['backbone_state'] and checkpoint['head_state']
    """

    experiment_mode, model_choice = infer_experiment_mode_and_model_choice(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )

    if "attention" not in experiment_mode:
        raise ValueError(f"Attention visualization requires an attention model, got experiment_mode='{experiment_mode}'.")

    is_vit = model_choice == "vit_base"

    print(f"📦 Loading model: experiment_mode=[{experiment_mode}] | backbone=[{model_choice}] | head=[ATTENTION]")

    # New unified checkpoint format
    if "model_state" in checkpoint:
        model = build_experiment_model(
            experiment_mode=experiment_mode,
            pretrained=False,
            num_attributes=len(ATTRIBUTE_NAMES),
            protest_num_outputs=1,
        ).to(device)

        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        if not hasattr(model, "backbone") or not hasattr(model, "head"):
            raise AttributeError("Unified model must expose .backbone and .head for attention visualization.")

        backbone = model.backbone
        head = model.head

        if not isinstance(head, CrossAttentionMultiTaskHead):
            raise TypeError(f"Expected CrossAttentionMultiTaskHead, got {type(head)}.")

        return model, backbone, head, model_choice, experiment_mode, is_vit

    # Legacy checkpoint format
    if "backbone_state" in checkpoint and "head_state" in checkpoint:
        backbone = MultiModelBackbone(
            model_name=model_choice,
            pretrained=False,
        ).to(device)

        head = CrossAttentionMultiTaskHead(
            input_dim=backbone.feature_dim,
            num_attributes=len(ATTRIBUTE_NAMES),
            protest_num_outputs=1,
        ).to(device)

        backbone.load_state_dict(checkpoint["backbone_state"], strict=True)
        head.load_state_dict(checkpoint["head_state"], strict=True)

        backbone.eval()
        head.eval()

        return None, backbone, head, model_choice, experiment_mode, is_vit

    raise KeyError("Checkpoint must contain either unified key 'model_state' or legacy keys 'backbone_state' and 'head_state'.")


def extract_weakly_supervised_bbox(
    checkpoint_path,
    raw_image_path,
    target_attribute="Sign",
    threshold_pct=0.15,
    output_dir="Visualize/attention",
):
    """
    Extract attention-derived coarse bounding boxes from task-query attention maps.

    This is qualitative weak-localization evidence, not formal localization proof.

    Supports:
        - unified checkpoints saved as {'model_state': ..., 'experiment_mode': ...}
        - legacy checkpoints saved as {'backbone_state': ..., 'head_state': ...}
    """

    if target_attribute not in ATTRIBUTE_NAMES:
        raise ValueError(f"Target attribute '{target_attribute}' is invalid. Choose from: {ATTRIBUTE_NAMES}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not os.path.exists(raw_image_path):
        raise FileNotFoundError(f"Image not found: {raw_image_path}")

    if not (0.0 < threshold_pct < 1.0):
        raise ValueError("threshold_pct must be between 0 and 1, e.g. 0.15 for top 15%.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model, backbone, head, model_choice, experiment_mode, is_vit = build_attention_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        device=device,
    )

    orig_image = Image.open(raw_image_path).convert("RGB")
    W_orig, H_orig = orig_image.size

    preprocess = get_protest_transforms(is_train=False, image_size=224)
    input_tensor = preprocess(orig_image).unsqueeze(0).to(device)

    target_attr_idx = ATTRIBUTE_NAMES.index(target_attribute)
    target_query_idx = 2 + target_attr_idx

    with torch.no_grad():
        features = backbone(input_tensor)

        if is_vit:
            if len(features.shape) != 3:
                raise ValueError(f"Expected ViT token tensor [B, N, C], got {features.shape}")

            spatial_features = features[:, 1:, :]
            num_tokens = spatial_features.shape[1]
            grid_size = int(num_tokens**0.5)

            if grid_size * grid_size != num_tokens:
                raise ValueError(f"Cannot reshape ViT tokens into square grid: num_tokens={num_tokens}")

            grid_h, grid_w = grid_size, grid_size

        else:
            if len(features.shape) != 4:
                raise ValueError(f"Expected CNN 4D feature map, got shape: {features.shape}")

            _, _, grid_h, grid_w = features.shape
            spatial_features = features.flatten(2).permute(0, 2, 1)

        queries = head.query_tokens.expand(1, -1, -1)

        _, attn_weights = head.cross_attn(
            query=queries,
            key=spatial_features,
            value=spatial_features,
            need_weights=True,
            average_attn_weights=False,
        )

        # Expected PyTorch shape with batch_first=True:
        #   [B, num_heads, num_queries, num_spatial_tokens]
        if attn_weights.ndim != 4:
            raise ValueError(f"Expected attention weights [B, H, Q, S], got shape={attn_weights.shape}")

        mean_attn = torch.mean(attn_weights, dim=1).squeeze(0)

        if target_query_idx >= mean_attn.shape[0]:
            raise IndexError(f"target_query_idx={target_query_idx} exceeds available queries={mean_attn.shape[0]}")

        attribute_attention_vector = mean_attn[target_query_idx].detach().cpu().numpy()

        predictions = head(features, is_vit=is_vit)
        predictions = standardize_outputs(predictions)

        attr_probs = torch.sigmoid(predictions["attribute_logits"])[0]
        target_prob = attr_probs[target_attr_idx].item()

    attn_matrix = attribute_attention_vector.reshape(grid_h, grid_w)
    attn_matrix = (attn_matrix - attn_matrix.min()) / (attn_matrix.max() - attn_matrix.min() + 1e-8)

    heatmap_resized = cv2.resize(
        attn_matrix,
        (W_orig, H_orig),
        interpolation=cv2.INTER_LINEAR,
    )

    cutoff_val = np.percentile(heatmap_resized, 100 * (1.0 - threshold_pct))
    binary_mask = np.where(heatmap_resized >= cutoff_val, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    img_with_box = np.array(orig_image).copy()
    box_coordinates = []

    min_area = max(100, int(0.0002 * W_orig * H_orig))

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            box_coordinates.append((x, y, w, h))
            cv2.rectangle(
                img_with_box,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3,
            )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_image)
    axes[0].set_title("1. Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title(
        f"2. Attention Mask: Top {int(threshold_pct * 100)}%",
        fontsize=11,
        fontweight="bold",
        color="darkorange",
    )
    axes[1].axis("off")

    axes[2].imshow(img_with_box)
    axes[2].set_title(
        f"3. Attention-Derived Box [{target_attribute.upper()}] | p={target_prob:.3f}",
        fontsize=11,
        fontweight="bold",
        color="green",
    )
    axes[2].axis("off")

    plt.tight_layout()

    safe_attribute_name = target_attribute.lower().replace(" ", "_")

    output_path = os.path.join(
        output_dir,
        f"bbox_proof_{experiment_mode}_{safe_attribute_name}.png",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Attention-derived bounding boxes computed successfully.")
    print(f"📊 Experiment mode: {experiment_mode}")
    print(f"📊 Backbone: {model_choice}")
    print(f"📊 Target attribute: {target_attribute} | probability={target_prob:.4f}")
    print(f"📊 Coordinates [x, y, width, height]: {box_coordinates}")
    print(f"🥇 Localization visualization exported to: {output_path}\n")

    return {
        "experiment_mode": experiment_mode,
        "model_choice": model_choice,
        "target_attribute": target_attribute,
        "target_probability": target_prob,
        "boxes": box_coordinates,
        "output_path": output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention-Derived Weak Localization Visualization Engine")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved PyTorch attention-head checkpoint.",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to target raw test image.",
    )

    parser.add_argument(
        "--attribute",
        type=str,
        default="Sign",
        choices=ATTRIBUTE_NAMES,
        help="Target attribute query.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Top attention fraction to keep, e.g. 0.15 means top 15%.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="Visualize/attention",
        help="Directory for exported localization figures.",
    )

    args = parser.parse_args()

    extract_weakly_supervised_bbox(
        checkpoint_path=args.checkpoint,
        raw_image_path=args.image,
        target_attribute=args.attribute,
        threshold_pct=args.threshold,
        output_dir=args.output_dir,
    )
