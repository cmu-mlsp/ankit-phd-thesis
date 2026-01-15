"""
Neural network models for ontological learning from weak labels.

This module contains Siamese network architectures with ontological layers
for hierarchical sound event classification.
"""

import torch
import torch.nn as nn


class BaseNet(nn.Module):
    """Base feature extraction network for AudioSet embeddings."""

    def __init__(self, in_features: int, hidden_dim: int = 512,
                 embedding_dim: int = 128, dropout: float = 0.5):
        """
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super(BaseNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class YamnetBaseNet(nn.Module):
    """Base feature extraction network for YAMNet embeddings."""

    def __init__(self, in_features: int, dropout: float = 0.3):
        """
        Args:
            in_features: Input feature dimension
            dropout: Dropout probability
        """
        super(YamnetBaseNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SiameseNet_OntologicalLayer(nn.Module):
    """
    Siamese network with ontological layer for hierarchical classification.

    Uses a shared base network to extract embeddings, then applies classification
    at two levels: subclass (level 1) and superclass (level 2) via an ontology matrix.
    """

    def __init__(self, num_classes_2: int, num_classes_1: int,
                 in_features: int, M):
        """
        Args:
            num_classes_2: Number of superclass (coarse) categories
            num_classes_1: Number of subclass (fine) categories
            in_features: Input feature dimension
            M: Ontology matrix mapping subclasses to superclasses (numpy array)
        """
        super(SiameseNet_OntologicalLayer, self).__init__()

        self.base = BaseNet(in_features)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, num_classes_1)
        self.bn = nn.BatchNorm1d(num_classes_1)
        self.sigmoid = nn.Sigmoid()

        # Fixed ontology layer (non-trainable)
        self.ontological_layer = nn.Linear(num_classes_1, num_classes_2, bias=False)
        self.ontological_layer.weight.requires_grad = False

        with torch.no_grad():
            self.ontological_layer.weight.copy_(torch.from_numpy(M))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass for Siamese network.

        Args:
            x1: First input batch
            x2: Second input batch (paired samples)

        Returns:
            Tuple of (embedding1, embedding2, out1_subclass, out1_superclass,
                     out2_subclass, out2_superclass)
        """
        # Shared feature extraction
        embedding1 = self.base(x1)
        embedding2 = self.base(x2)

        # Subclass predictions
        out1_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding1))))
        out2_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding2))))

        # Convert to probabilities
        out1_1_prob = self.sigmoid(out1_1)
        out2_1_prob = self.sigmoid(out2_1)

        # Superclass predictions via ontology layer
        out1_2 = self.ontological_layer(out1_1_prob)
        out2_2 = self.ontological_layer(out2_1_prob)

        return embedding1, embedding2, out1_1, out1_2, out2_1, out2_2


class YamnetSiameseNet_OntologicalLayer(nn.Module):
    """
    Siamese network with ontological layer for YAMNet embeddings.

    Similar to SiameseNet_OntologicalLayer but optimized for YAMNet features.
    """

    def __init__(self, num_classes_2: int, num_classes_1: int,
                 in_features: int, M):
        """
        Args:
            num_classes_2: Number of superclass (coarse) categories
            num_classes_1: Number of subclass (fine) categories
            in_features: Input feature dimension
            M: Ontology matrix mapping subclasses to superclasses (numpy array)
        """
        super(YamnetSiameseNet_OntologicalLayer, self).__init__()

        self.base = YamnetBaseNet(in_features)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, num_classes_1)
        self.bn = nn.BatchNorm1d(num_classes_1)
        self.sigmoid = nn.Sigmoid()

        # Fixed ontology layer (non-trainable)
        self.ontological_layer = nn.Linear(num_classes_1, num_classes_2, bias=False)
        self.ontological_layer.weight.requires_grad = False

        with torch.no_grad():
            self.ontological_layer.weight.copy_(torch.from_numpy(M))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass for Siamese network with YAMNet features.

        Args:
            x1: First input batch
            x2: Second input batch (paired samples)

        Returns:
            Tuple of (embedding1, embedding2, out1_subclass, out1_superclass,
                     out2_subclass, out2_superclass)
        """
        # Shared feature extraction
        embedding1 = self.base(x1)
        embedding2 = self.base(x2)

        # Subclass predictions
        out1_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding1))))
        out2_1 = self.bn(self.output_layer(self.dropout(self.relu(embedding2))))

        # Convert to probabilities
        out1_1_prob = self.sigmoid(out1_1)
        out2_1_prob = self.sigmoid(out2_1)

        # Superclass predictions via ontology layer
        out1_2 = self.ontological_layer(out1_1_prob)
        out2_2 = self.ontological_layer(out2_1_prob)

        return embedding1, embedding2, out1_1, out1_2, out2_1, out2_2
