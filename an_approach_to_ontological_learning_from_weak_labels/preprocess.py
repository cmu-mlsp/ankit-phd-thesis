"""
Data preprocessing utilities for AudioSet with ontology information.

Loads AudioSet data from TFRecord files and organizes ontology information
from JSON files to obtain hierarchical class labels.
"""

import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset


def get_all_children(category: str, aso: Dict) -> List[Dict]:
    """
    Recursively get all children of a category in the ontology.

    Args:
        category: Category ID to get children for
        aso: AudioSet ontology dictionary

    Returns:
        List of child category dictionaries with names and nested children
    """
    childs = aso[category]["child_ids"]
    childs_names = []

    for child in childs:
        child_name = {"name": aso[child]["name"]}
        if "child_ids" in aso[child]:
            child_name["children"] = get_all_children(child, aso)
        childs_names.append(child_name)

    return childs_names if childs_names else None


def preprocess_ontology(data_dir: str) -> Tuple[Dict, List[str]]:
    """
    Load and preprocess AudioSet ontology from JSON file.

    Args:
        data_dir: Directory containing ontology.json

    Returns:
        Tuple of (ontology_dict, list_of_root_category_ids)
    """
    with open(os.path.join(data_dir, 'ontology.json')) as f:
        ont_data = json.load(f)

    ont = {}
    for category in ont_data:
        ont[category["id"]] = {
            "name": category["name"],
            "restrictions": category["restrictions"],
            "child_ids": category["child_ids"],
            "parents_ids": []
        }

    # Build parent relationships
    for category in ont:
        for child in ont[category]["child_ids"]:
            ont[child]["parents_ids"].append(category)

    # Find root categories (no parents)
    higher_categories = [
        cat_id for cat_id, cat_data in ont.items()
        if not cat_data["parents_ids"]
    ]

    return ont, higher_categories


def get_all_parents(id_: str, ont: Dict, parents_names: List[str]) -> None:
    """
    Recursively collect all parent category names up to root.

    Args:
        id_: Category ID to get parents for
        ont: Ontology dictionary
        parents_names: List to append parent names to (modified in place)
    """
    parents = ont[id_]["parents_ids"]

    if parents:
        for parent in parents:
            get_all_parents(parent, ont, parents_names)

    parents_names.append(ont[id_]["name"])


def extract_tfrecord_data(data_dir: str, features_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract audio features and labels from TFRecord files.

    Args:
        data_dir: Base directory containing metadata files
        features_dir: Subdirectory containing TFRecord files

    Returns:
        Tuple of (audio_features, subclass_indices, superclass_indices)
    """
    # Load class label mappings
    with open(os.path.join(data_dir, 'class_labels_indices.csv'), mode='r') as file:
        reader = csv.reader(file)
        index_to_id = {rows[0]: rows[1] for rows in reader}

    tfrecord_files = os.listdir(os.path.join(data_dir, features_dir))

    sounds_data = []
    class_1 = []
    class_2 = []

    context_description = {"video_id": "byte", "labels": "int"}
    sequence_description = {"audio_embedding": "byte"}

    ont, _ = preprocess_ontology(data_dir)
    count = 0

    for filename in tfrecord_files:
        tfrecord_path = os.path.join(data_dir, features_dir, filename)
        dataset = TFRecordDataset(
            tfrecord_path, index_path=None,
            description=context_description,
            sequence_description=sequence_description
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for data in iter(loader):
            data_labels = [int(i) for i in data[0]['labels'].tolist()[0]]

            # Collect hierarchical class labels
            parent_names = []
            for i in data[0]['labels'][0].numpy():
                get_all_parents(index_to_id[str(i)], ont, parent_names)

            # Only include samples with at least 2 hierarchy levels
            if len(parent_names) > 1:
                file_data = [t.numpy() for t in data[1]['audio_embedding']]
                class_1.append(parent_names[0])
                class_2.append(parent_names[1])
                sounds_data.append(np.concatenate(file_data))
                count += 1

    print(f"Extracted {count} samples")

    sounds_data = np.asanyarray(sounds_data, dtype=object)

    # Convert class names to indices
    unique_class1 = np.unique(class_1)
    unique_class2 = np.unique(class_2)

    class1_index = np.array([
        np.where(class_1[i] == unique_class1)[0][0]
        for i in range(len(class_1))
    ], dtype=int)

    class2_index = np.array([
        np.where(class_2[i] == unique_class2)[0][0]
        for i in range(len(class_2))
    ], dtype=int)

    return sounds_data, class1_index, class2_index


def class_to_index(data_dir: str, features_dir: str) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Build class name to index mappings from TFRecord files.

    Args:
        data_dir: Base directory containing metadata files
        features_dir: Subdirectory containing TFRecord files

    Returns:
        Tuple of (subclass_to_idx, superclass_to_idx, unique_subclasses, unique_superclasses)
    """
    with open(os.path.join(data_dir, 'class_labels_indices.csv'), mode='r') as file:
        reader = csv.reader(file)
        index_to_id = {rows[0]: rows[1] for rows in reader}

    tfrecord_files = os.listdir(os.path.join(data_dir, features_dir))

    class_1 = []
    class_2 = []

    context_description = {"video_id": "byte", "labels": "int"}
    sequence_description = {"audio_embedding": "byte"}

    ont, highest_category = preprocess_ontology(data_dir)
    print(f"Root categories: {highest_category}")

    for filename in tfrecord_files:
        tfrecord_path = os.path.join(data_dir, features_dir, filename)
        dataset = TFRecordDataset(
            tfrecord_path, index_path=None,
            description=context_description,
            sequence_description=sequence_description
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for data in iter(loader):
            parent_names = []
            for i in data[0]['labels'][0].numpy():
                get_all_parents(index_to_id[str(i)], ont, parent_names)
            parent_names = np.unique(parent_names)

            if len(parent_names) > 1:
                class_1.append(parent_names[0])
                class_2.append(parent_names[1])

    unique_class1 = np.unique(class_1)
    unique_class2 = np.unique(class_2)

    class_to_index_1 = {name: i for i, name in enumerate(unique_class1)}
    class_to_index_2 = {name: i for i, name in enumerate(unique_class2)}

    return class_to_index_1, class_to_index_2, unique_class1, unique_class2


def get_child_index(ont: Dict, class_to_index_1: Dict, class_to_index_2: Dict,
                    unique_class_1: np.ndarray, data_dir: str) -> np.ndarray:
    """
    Build ontology matrix mapping subclasses to superclasses.

    Args:
        ont: Ontology dictionary
        class_to_index_1: Subclass name to index mapping
        class_to_index_2: Superclass name to index mapping
        unique_class_1: Array of unique subclass names
        data_dir: Directory containing ontology.json

    Returns:
        Binary matrix where M[i,j]=1 if subclass i belongs to superclass j
    """
    with open(os.path.join(data_dir, 'ontology.json')) as f:
        ont_data = json.load(f)

    with open(os.path.join(data_dir, 'class_labels_indices.csv'), mode='r') as file:
        reader = csv.reader(file)
        class_to_id = {rows[2]: rows[1] for rows in reader}

    ont_matrix = np.zeros((len(class_to_index_1), len(class_to_index_2)))

    for class_ in unique_class_1:
        # Find category ID for this class
        id_ = None
        for cat in ont_data:
            if cat["name"] == class_:
                id_ = cat["id"]
                break

        if id_ is None:
            continue

        # Mark all children in the ontology matrix
        children = ont[id_]['child_ids']
        for child in children:
            class_2_name = ont[child]['name']
            if class_2_name in class_to_index_2:
                ont_matrix[class_to_index_1[class_]][class_to_index_2[class_2_name]] = 1

    return ont_matrix


if __name__ == "__main__":
    data_dir = './'
    train_features_dir = 'audioset_v1_embeddings/bal_train/'

    class_to_index_1, class_to_index_2, unique_class_1, unique_class_2 = class_to_index(
        data_dir, train_features_dir
    )
    print(f"Superclass mapping: {class_to_index_2}")

    ont, _ = preprocess_ontology(data_dir)
    ont_matrix = get_child_index(
        ont, class_to_index_1, class_to_index_2,
        unique_class_1, data_dir
    )
