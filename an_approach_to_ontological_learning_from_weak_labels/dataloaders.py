"""
Dataset classes for Siamese network training with ontological learning.

These datasets generate pairs of audio samples for training Siamese networks
with hierarchical (ontological) supervision at multiple levels.
"""

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class AudioSet_Siamese(Dataset):
    """
    Dataset for Siamese network training on AudioSet with weak labels.

    Generates pairs of samples with three types:
    - Type 0: Same subclass (most similar)
    - Type 1: Different subclass, same superclass (moderately similar)
    - Type 2: Different superclass (dissimilar)
    """

    def __init__(self, data, labels1, labels2,
                 num_subclasses: int, num_superclasses: int,
                 seg_per_clip: int = 10):
        """
        Args:
            data: List of audio feature arrays (one per clip)
            labels1: Subclass labels for each clip
            labels2: Superclass labels for each clip
            num_subclasses: Total number of subclass categories
            num_superclasses: Total number of superclass categories
            seg_per_clip: Number of segments per audio clip
        """
        self.data = np.concatenate(data, axis=0).astype('float')
        self.seg_per_clip = seg_per_clip
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses

        num_clips = len(data)

        # Build one-hot encoded label matrices
        self.logits_1 = np.zeros((num_clips, num_subclasses), dtype=np.int64)
        self.logits_2 = np.zeros((num_clips, num_superclasses), dtype=np.int64)

        for i in range(num_clips):
            self.logits_1[i][self.labels1[i].astype('int')] = 1
            self.logits_2[i][self.labels2[i].astype('int')] = 1

        # Precompute indices for efficient pair sampling
        self.same_subclass = [
            np.where(np.all(self.logits_1[i] == self.logits_1, axis=1))[0]
            for i in range(num_clips)
        ]

        self.diff_subclass_same_superclass = [
            np.where(
                np.logical_and(
                    np.logical_not(np.all(self.logits_1[i] == self.logits_1, axis=1)),
                    np.all(self.logits_2[i] == self.logits_2, axis=1)
                )
            )[0]
            for i in range(num_clips)
        ]

        self.diff_superclass = [
            np.where(np.logical_not(np.all(self.logits_2[i] == self.logits_2, axis=1)))[0]
            for i in range(num_clips)
        ]

    def __len__(self) -> int:
        return int(self.length * 5)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a pair of samples with their labels and pair type."""
        idx = np.random.choice(np.arange(self.length))
        class_idx = int(idx / self.seg_per_clip)

        # Randomly select pair type with equal probability
        class2_type = random.uniform(0, 1)

        if class2_type < 1/3:
            # Same subclass
            x2_idxs = self.same_subclass[class_idx]
            x2_idx = np.random.choice(x2_idxs)
            pair_type = 0

        elif class2_type < 2/3:
            # Same superclass, different subclass
            x2_idxs = self.diff_subclass_same_superclass[class_idx]
            if x2_idxs.size != 0:
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 1
            else:
                # Fallback to different superclass if no valid pair exists
                x2_idxs = self.diff_superclass[class_idx]
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 2

        else:
            # Different superclass
            x2_idxs = self.diff_superclass[class_idx]
            x2_idx = np.random.choice(x2_idxs)
            pair_type = 2

        x1 = self.data[idx]

        # Get second sample from paired clip
        x2_train_idx = x2_idx * self.seg_per_clip + np.random.choice(np.arange(10))
        x2 = self.data[x2_train_idx]

        # Get labels
        x1_superclass = self.logits_2[class_idx]
        x2_superclass = self.logits_2[x2_idx]
        x1_subclass = self.logits_1[class_idx]
        x2_subclass = self.logits_1[x2_idx]

        return (x1.flatten(), x2.flatten(),
                x1_subclass, x1_superclass,
                x2_subclass, x2_superclass, pair_type)


class AudioSet_Siamese_Eval(Dataset):
    """Dataset for evaluating Siamese networks on AudioSet."""

    def __init__(self, data, labels1, labels2,
                 num_subclasses: int, num_superclasses: int,
                 seg_per_clip: int = 10):
        """
        Args:
            data: List of audio feature arrays (one per clip)
            labels1: Subclass labels for each clip
            labels2: Superclass labels for each clip
            num_subclasses: Total number of subclass categories
            num_superclasses: Total number of superclass categories
            seg_per_clip: Number of segments per audio clip
        """
        self.data = np.concatenate(data, axis=0).astype('float')
        self.seg_per_clip = seg_per_clip
        self.length = self.data.shape[0]
        self.labels1 = labels1
        self.labels2 = labels2
        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses

        num_clips = len(data)

        self.logits_1 = np.zeros((num_clips, num_subclasses), dtype=np.int64)
        self.logits_2 = np.zeros((num_clips, num_superclasses), dtype=np.int64)

        for i in range(num_clips):
            self.logits_1[i][self.labels1[i].astype('int')] = 1
            self.logits_2[i][self.labels2[i].astype('int')] = 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        """Get a single sample with its labels."""
        x1 = self.data[index]
        clip_idx = int(index / self.seg_per_clip)
        return x1.flatten(), self.logits_1[clip_idx], self.logits_2[clip_idx]


class AudioSet_Strong_Siamese(Dataset):
    """
    Dataset for Siamese network training on AudioSet with strong labels.

    Similar to AudioSet_Siamese but handles segment-level (strong) labels
    rather than clip-level (weak) labels.
    """

    def __init__(self, data, labels1, labels2,
                 num_subclasses: int, num_superclasses: int):
        """
        Args:
            data: List of audio feature arrays
            labels1: Subclass labels (segment-level)
            labels2: Superclass labels (segment-level)
            num_subclasses: Total number of subclass categories
            num_superclasses: Total number of superclass categories
        """
        self.data = np.concatenate(data, axis=0).astype('float')
        self.seg_per_clip = 10
        self.length = self.data.shape[0]

        # Handle both clip-level and segment-level labels
        self.labels1 = labels1
        if len(data) == len(labels1):
            self.labels1 = np.concatenate(labels1, axis=0)
        self.labels1 = [np.asarray(lbl) for lbl in self.labels1]

        self.labels2 = labels2
        if len(data) == len(labels2):
            self.labels2 = np.concatenate(labels2, axis=0)
        self.labels2 = [np.asarray(lbl) for lbl in self.labels2]

        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses

        num_clips = len(self.data)

        # Build one-hot encoded label matrices
        self.logits_1 = np.zeros((num_clips, num_subclasses), dtype=np.int64)
        self.logits_2 = np.zeros((num_clips, num_superclasses), dtype=np.int64)

        for i in range(num_clips):
            self.logits_1[i][self.labels1[i]] = 1
            self.logits_2[i][self.labels2[i]] = 1

    def __len__(self) -> int:
        return int(self.length * 5)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a pair of samples with their labels and pair type."""
        idx = np.random.choice(np.arange(self.length))

        class2_type = random.uniform(0, 1)

        if class2_type < 1/3:
            # Same subclass
            x2_idxs = np.where(
                np.logical_and(
                    np.all(self.logits_1[idx] == self.logits_1, axis=1),
                    np.all(self.logits_2[idx] == self.logits_2, axis=1)
                )
            )[0]
            x2_idx = np.random.choice(x2_idxs)
            pair_type = 0

        elif class2_type < 2/3:
            # Same superclass, different subclass (no overlap in subclasses)
            x2_idxs = np.where(
                np.logical_and(
                    np.sum(self.logits_1[idx] != self.logits_1, axis=1) ==
                    np.sum(self.logits_1[idx]) + np.sum(self.logits_1, axis=1),
                    np.all(self.logits_2[idx] == self.logits_2, axis=1)
                )
            )[0]

            if x2_idxs.size != 0:
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 1
            else:
                # Fallback to different superclass
                x2_idxs = np.where(
                    np.logical_not(np.all(self.logits_2[idx] == self.logits_2, axis=1))
                )[0]
                x2_idx = np.random.choice(x2_idxs)
                pair_type = 2

        else:
            # Different superclass
            x2_idxs = np.where(
                np.logical_not(np.all(self.logits_2[idx] == self.logits_2, axis=1))
            )[0]
            x2_idx = np.random.choice(x2_idxs)
            pair_type = 2

        x1 = self.data[idx]
        x2 = self.data[x2_idx]

        x1_superclass = self.logits_2[idx]
        x2_superclass = self.logits_2[x2_idx]
        x1_subclass = self.logits_1[idx]
        x2_subclass = self.logits_1[x2_idx]

        return (x1.flatten(), x2.flatten(),
                x1_subclass, x1_superclass,
                x2_subclass, x2_superclass, pair_type)


class AudioSet_Strong_Siamese_Eval(Dataset):
    """Dataset for evaluating Siamese networks on AudioSet with strong labels."""

    def __init__(self, data, labels1, labels2,
                 num_subclasses: int, num_superclasses: int):
        """
        Args:
            data: List of audio feature arrays
            labels1: Subclass labels (segment-level)
            labels2: Superclass labels (segment-level)
            num_subclasses: Total number of subclass categories
            num_superclasses: Total number of superclass categories
        """
        self.data = np.concatenate(data, axis=0).astype('float')
        self.seg_per_clip = 10
        self.length = self.data.shape[0]

        self.labels1 = np.concatenate(labels1, axis=0)
        self.labels1 = [np.asarray(lbl) for lbl in self.labels1]

        self.labels2 = np.concatenate(labels2, axis=0)
        self.labels2 = [np.asarray(lbl) for lbl in self.labels2]

        self.num_subclasses = num_subclasses
        self.num_superclasses = num_superclasses

        num_clips = self.data.shape[0]

        self.logits_1 = np.zeros((num_clips, num_subclasses), dtype=np.int64)
        self.logits_2 = np.zeros((num_clips, num_superclasses), dtype=np.int64)

        for i in range(num_clips):
            self.logits_1[i][self.labels1[i]] = 1
            self.logits_2[i][self.labels2[i]] = 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        """Get a single sample with its labels."""
        x1 = self.data[index]
        return x1.flatten(), self.logits_1[index], self.logits_2[index]
