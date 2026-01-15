"""
Loss functions for ontological learning with Siamese networks.

Combines classification losses at multiple hierarchy levels with
contrastive embedding loss based on ontological distance.
"""

import torch
import torch.nn as nn


class Ontological_Loss(nn.Module):
    """
    Multi-task loss for ontological Siamese network training.

    Combines:
    - Binary cross-entropy loss for subclass (level 1) classification
    - Binary cross-entropy loss for superclass (level 2) classification
    - Contrastive loss based on ontological distance in embedding space
    """

    def __init__(self, lambda1: float, lambda2: float, lambda3: float,
                 weights_1: torch.Tensor, weights_2: torch.Tensor):
        """
        Args:
            lambda1: Weight for subclass classification loss
            lambda2: Weight for superclass classification loss
            lambda3: Weight for contrastive embedding loss
            weights_1: Class weights for subclass BCE loss (handles class imbalance)
            weights_2: Class weights for superclass BCE loss
        """
        super(Ontological_Loss, self).__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.weights_1 = weights_1
        self.weights_2 = weights_2

    def forward(self, out, targets, pair_type: torch.Tensor) -> torch.Tensor:
        """
        Compute combined ontological loss.

        Args:
            out: Model outputs (embedding1, embedding2, out1_1, out1_2, out2_1, out2_2)
            targets: Ground truth labels (target1_1, target1_2, target2_1, target2_2)
            pair_type: Type of pair (0=same subclass, 1=same superclass, 2=different)

        Returns:
            Combined loss value
        """
        embedding1, embedding2, out1_1, out1_2, out2_1, out2_2 = out
        target1_1, target1_2, target2_1, target2_2 = targets

        # Contrastive target: scales with pair type (0, 2, 4)
        ont_target = pair_type * 2

        # Subclass classification loss
        loss1 = nn.BCEWithLogitsLoss(pos_weight=self.weights_1)
        L1_1 = loss1(out1_1, target1_1.float())
        L1_2 = loss1(out2_1, target2_1.float())

        # Superclass classification loss
        # Convert probabilities back to logits for BCEWithLogitsLoss
        loss2 = nn.BCEWithLogitsLoss(pos_weight=self.weights_2)
        out1_2_logits = -torch.log((1 / out1_2) - 1)
        out2_2_logits = -torch.log((1 / out2_2) - 1)
        L2_1 = loss2(out1_2_logits, target1_2.float())
        L2_2 = loss2(out2_2_logits, target2_2.float())

        # Contrastive loss: embedding distance should match ontological distance
        D_w = (torch.sqrt(torch.sum((embedding1 - embedding2)**2, axis=1)) - ont_target)**2

        loss = (self.lambda1 * (L1_1 + L1_2) +
                self.lambda2 * (L2_1 + L2_2) +
                self.lambda3 * D_w)

        return loss.mean()


class Ontological_Loss_Unweighted(nn.Module):
    """
    Unweighted version of Ontological_Loss.

    Same as Ontological_Loss but without class weights for BCE losses.
    Suitable when class imbalance is not a significant concern.
    """

    def __init__(self, lambda1: float, lambda2: float, lambda3: float):
        """
        Args:
            lambda1: Weight for subclass classification loss
            lambda2: Weight for superclass classification loss
            lambda3: Weight for contrastive embedding loss
        """
        super(Ontological_Loss_Unweighted, self).__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, out, targets, pair_type: torch.Tensor) -> torch.Tensor:
        """
        Compute combined ontological loss without class weights.

        Args:
            out: Model outputs (embedding1, embedding2, out1_1, out1_2, out2_1, out2_2)
            targets: Ground truth labels (target1_1, target1_2, target2_1, target2_2)
            pair_type: Type of pair (0=same subclass, 1=same superclass, 2=different)

        Returns:
            Combined loss value
        """
        embedding1, embedding2, out1_1, out1_2, out2_1, out2_2 = out
        target1_1, target1_2, target2_1, target2_2 = targets

        ont_target = pair_type * 2

        # Subclass classification loss (with logits)
        loss1 = nn.BCEWithLogitsLoss()
        L1_1 = loss1(out1_1, target1_1.float())
        L1_2 = loss1(out2_1, target2_1.float())

        # Superclass classification loss (outputs are already probabilities)
        loss2 = nn.BCELoss()
        L2_1 = loss2(out1_2, target1_2.float())
        L2_2 = loss2(out2_2, target2_2.float())

        # Contrastive loss
        D_w = (torch.sqrt(torch.sum((embedding1 - embedding2)**2, axis=1)) - ont_target)**2

        loss = (self.lambda1 * (L1_1 + L1_2) +
                self.lambda2 * (L2_1 + L2_2) +
                self.lambda3 * D_w)

        return loss.mean()
