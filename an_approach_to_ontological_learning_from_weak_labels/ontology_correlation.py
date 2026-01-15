"""
Generate ontology correlation matrices for the AudioSet hierarchy.

Creates two types of correlation matrices:
1. Method 1: Subclass-to-subclass correlation (same parent = correlated)
2. Method 2: Full hierarchy correlation including super-sub relationships
"""

import numpy as np

# Ontology layer: mapping from 43 subclasses to 7 superclasses
# Each row is a superclass, each column is a subclass
# M[i,j] = 1 if subclass j belongs to superclass i
ONTOLOGY_MATRIX = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

SAVE_DIR = 'data/'


def compute_subclass_correlation(M: np.ndarray) -> np.ndarray:
    """
    Compute subclass-to-subclass correlation based on shared parent.

    Two subclasses are correlated (1) if they share the exact same
    set of parent superclasses.

    Args:
        M: Ontology matrix (num_subclasses x num_superclasses)

    Returns:
        Binary correlation matrix (num_subclasses x num_subclasses)
    """
    num_lower = M.shape[0]
    correlation = np.zeros((num_lower, num_lower), dtype=np.int64)

    for i in range(num_lower):
        for j in range(num_lower):
            if np.all(M[i] == M[j]):
                correlation[i, j] = 1

    return correlation


def compute_hierarchy_correlation(M: np.ndarray) -> np.ndarray:
    """
    Compute full hierarchy correlation including super-sub relationships.

    Creates a combined correlation matrix that includes:
    - Superclass to superclass identity
    - Superclass to subclass membership
    - Subclass to superclass membership (transpose)

    Args:
        M: Ontology matrix (num_subclasses x num_superclasses)

    Returns:
        Correlation matrix (num_total x num_total) where
        num_total = num_subclasses + num_superclasses
    """
    num_lower, num_upper = M.shape
    num_all = num_lower + num_upper

    correlation = np.zeros((num_all, num_all), dtype=np.int64)

    # Pad ontology matrix and place in upper rows
    M_pad = np.pad(M.T, ((0, 0), (num_upper, 0)), 'constant', constant_values=0)
    correlation[:num_upper] = M_pad

    # Make symmetric
    correlation = correlation + correlation.T

    return correlation


def main():
    """Generate and save both correlation matrices."""
    # Transpose to get (num_subclasses x num_superclasses)
    M = ONTOLOGY_MATRIX.T
    num_lower, num_upper = M.shape

    print(f"Ontology matrix shape: {num_lower} subclasses x {num_upper} superclasses")

    # Method 1: Subclass correlation based on shared parents
    correlation_1 = compute_subclass_correlation(M)
    np.save(SAVE_DIR + 'ontology_correlation_method_1.npy', correlation_1)
    print(f"Saved subclass correlation matrix: {correlation_1.shape}")

    # Method 2: Full hierarchy correlation
    correlation_2 = compute_hierarchy_correlation(M)
    np.save(SAVE_DIR + 'ontology_correlation_method_2.npy', correlation_2)
    print(f"Saved hierarchy correlation matrix: {correlation_2.shape}")


if __name__ == "__main__":
    main()
