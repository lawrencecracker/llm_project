"""
Data utility functions for loading and preprocessing datasets
"""

import json
from pathlib import Path
from typing import Union, Dict
from datasets import load_dataset, Dataset
import logging

logger = logging.getLogger(__name__)


def load_dataset_from_json(file_path: str) -> Dataset:
    """Load dataset from JSONL file"""
    logger.info(f"Loading dataset from {file_path}")
    
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    dataset = Dataset.from_dict({
        "text": [item.get("text", str(item)) for item in data]
    })
    return dataset


def load_dataset_from_csv(file_path: str, text_column: str = "text") -> Dataset:
    """Load dataset from CSV file"""
    logger.info(f"Loading dataset from {file_path}")
    
    import csv
    data = {"text": []}
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["text"].append(row.get(text_column, ""))
    
    dataset = Dataset.from_dict(data)
    return dataset


def load_dataset_from_txt(file_path: str, split_by: str = "\n\n") -> Dataset:
    """Load dataset from text file"""
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    documents = content.split(split_by)
    
    dataset = Dataset.from_dict({
        "text": [doc.strip() for doc in documents if doc.strip()]
    })
    return dataset


def load_hf_dataset(dataset_name: str, config: str = None, split: str = "train") -> Dataset:
    """Load dataset from Hugging Face Hub"""
    logger.info(f"Loading {split} split of {dataset_name}")
    dataset = load_dataset(dataset_name, config, split=split)
    return dataset


def combine_datasets(datasets: list) -> Dataset:
    """Combine multiple datasets"""
    logger.info(f"Combining {len(datasets)} datasets")
    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = combined.concatenate(dataset)
    return combined
