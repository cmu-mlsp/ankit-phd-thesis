"""
Evaluation functions for ontological Siamese networks.

Computes mean Average Precision (mAP) and AUC metrics at both
subclass and superclass hierarchy levels.
"""

from typing import Tuple

import numpy as np
import torch
from sklearn import metrics


def evaluate_model_stats(data_loader, model, device: torch.device = None,
                         skip_classes: list = None) -> Tuple[float, float, float, float]:
    """
    Evaluate model performance on a dataset.

    Args:
        data_loader: DataLoader for evaluation dataset
        model: Trained Siamese network model
        device: Device to run on (default: cuda:0)
        skip_classes: List of subclass indices to skip in evaluation
                      (e.g., classes not present in validation set)

    Returns:
        Tuple of (mAP_subclass, AUC_subclass, mAP_superclass, AUC_superclass)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if skip_classes is None:
        skip_classes = []

    model.eval()

    complete_outputs_1 = []
    complete_targets_1 = []
    complete_outputs_2 = []
    complete_targets_2 = []

    with torch.no_grad():
        for input1, target1_1, target1_2 in data_loader:
            # Move to device
            input1 = input1.to(device).float()
            target1_1 = target1_1.to(device)
            target1_2 = target1_2.to(device)

            batch_size = int(input1.shape[0] / 2)

            # Model forward pass (using Siamese structure)
            _, _, out1_1, out1_2, out2_1, out2_2 = model.forward(
                input1[0:batch_size], input1[batch_size:]
            )

            # Convert logits to probabilities
            sigmoid = torch.nn.Sigmoid()
            out1_1 = sigmoid(out1_1)
            out2_1 = sigmoid(out2_1)

            complete_outputs_1.append(torch.cat((out1_1, out2_1)))
            complete_targets_1.append(target1_1)

            complete_outputs_2.append(torch.cat((out1_2, out2_2)))
            complete_targets_2.append(target1_2)

    # Concatenate all batches
    complete_outputs_1 = torch.cat(complete_outputs_1, 0)
    complete_targets_1 = torch.cat(complete_targets_1, 0)
    complete_outputs_2 = torch.cat(complete_outputs_2, 0)
    complete_targets_2 = torch.cat(complete_targets_2, 0)

    return compute_stats(
        complete_outputs_1, complete_targets_1,
        complete_outputs_2, complete_targets_2,
        skip_classes=skip_classes
    )


def compute_stats(output_1: torch.Tensor, target_1: torch.Tensor,
                  output_2: torch.Tensor, target_2: torch.Tensor,
                  skip_classes: list = None,
                  segments_per_clip: int = 10) -> Tuple[float, float, float, float]:
    """
    Compute mAP and AUC metrics at both hierarchy levels.

    Args:
        output_1: Model predictions for subclass level
        target_1: Ground truth labels for subclass level
        output_2: Model predictions for superclass level
        target_2: Ground truth labels for superclass level
        skip_classes: List of subclass indices to skip
        segments_per_clip: Number of segments to average per clip

    Returns:
        Tuple of (mAP_subclass, AUC_subclass, mAP_superclass, AUC_superclass)
    """
    if skip_classes is None:
        skip_classes = []

    num_classes_1 = output_1.shape[-1]
    num_classes_2 = output_2.shape[-1]

    # Move to CPU for sklearn metrics
    target_1 = target_1.detach().cpu().numpy()
    output_1 = output_1.detach().cpu().numpy()
    target_2 = target_2.detach().cpu().numpy()
    output_2 = output_2.detach().cpu().numpy()

    # Average predictions across segments within each clip
    output_1 = np.mean(
        output_1.reshape(segments_per_clip, -1, num_classes_1), axis=0
    )
    output_2 = np.mean(
        output_2.reshape(segments_per_clip, -1, num_classes_2), axis=0
    )

    # Compute subclass (level 1) metrics
    average_precision_1 = np.zeros((num_classes_1,))
    auc_1 = np.zeros((num_classes_1,))

    valid_classes_1 = 0
    for i in range(num_classes_1):
        if i not in skip_classes:
            average_precision_1[i] = metrics.average_precision_score(
                target_1[:, i], output_1[:, i]
            )
            auc_1[i] = metrics.roc_auc_score(
                target_1[:, i], output_1[:, i], average=None
            )
            valid_classes_1 += 1

    # Compute superclass (level 2) metrics
    average_precision_2 = np.zeros((num_classes_2,))
    auc_2 = np.zeros((num_classes_2,))

    for i in range(num_classes_2):
        average_precision_2[i] = metrics.average_precision_score(
            target_2[:, i], output_2[:, i]
        )
        auc_2[i] = metrics.roc_auc_score(
            target_2[:, i], output_2[:, i], average=None
        )

    # Compute mean metrics
    mAP_1 = np.sum(average_precision_1) / valid_classes_1 if valid_classes_1 > 0 else 0
    auc_1_mean = np.sum(auc_1) / valid_classes_1 if valid_classes_1 > 0 else 0

    mAP_2 = np.mean(average_precision_2)
    auc_2_mean = np.mean(auc_2)

    return mAP_1, auc_1_mean, mAP_2, auc_2_mean
