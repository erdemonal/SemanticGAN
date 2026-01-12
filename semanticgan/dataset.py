"""
Dataset loader for knowledge graph triples.

This module provides a PyTorch Dataset class for loading knowledge graph triples
from tab-separated files and preparing them for training.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import json


class KnowledgeGraphDataset(Dataset):
    """
    PyTorch Dataset for knowledge graph triples.
    
    Loads knowledge graph triples from a tab-separated file and optionally
    loads entity/relation mappings from a JSON file. Supports flexible
    column naming and separator specification.
    
    Args:
        triples_path (str): Path to the triples file (tab-separated by default).
        mappings_path (str, optional): Path to JSON file containing entity/relation mappings.
            If provided, entity and relation counts are loaded from this file.
            If None, counts are inferred from the data. Default: None.
        sep (str): Column separator in the triples file. Default: '\\t'.
        header (int, optional): Row number to use as column names. None if no header. Default: None.
        names (list): Column names for the triples file. Default: ['h', 'r', 't'].
            Expected format: [head_column, relation_column, tail_column].
    
    Attributes:
        num_entities (int): Number of unique entities in the dataset.
        num_relations (int): Number of unique relations in the dataset.
        head (torch.Tensor): Head entity IDs.
        rel (torch.Tensor): Relation IDs.
        tail (torch.Tensor): Tail entity IDs.
        data (torch.Tensor): Stacked tensor of shape (num_triples, 3).
    
    Example:
        >>> dataset = KnowledgeGraphDataset(
        ...     triples_path="data/kg_triples.txt",
        ...     mappings_path="data/mappings.json",
        ...     sep='\\t',
        ...     names=['h', 'r', 't']
        ... )
        >>> print(f"Loaded {len(dataset)} triples")
        >>> print(f"Entities: {dataset.num_entities}, Relations: {dataset.num_relations}")
        >>> sample = dataset[0]
        >>> print(sample)  # {'head': tensor(123), 'relation': tensor(5), 'tail': tensor(456)}
    """
    
    def __init__(self, triples_path, mappings_path=None, sep='\t', header=None, names=['h', 'r', 't']):
        print(f"Loading dataset from {triples_path}...")
        
        try:
            df = pd.read_csv(
                triples_path, 
                sep=sep, 
                header=header, 
                names=names, 
                dtype={names[0]: np.int32, names[1]: np.int32, names[2]: np.int32},
                engine='c' 
            )
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise e

        self.head = torch.tensor(df[names[0]].values, dtype=torch.long)
        self.rel = torch.tensor(df[names[1]].values, dtype=torch.long)
        self.tail = torch.tensor(df[names[2]].values, dtype=torch.long)
        self.data = torch.stack([self.head, self.rel, self.tail], dim=1)
        
        self.num_entities = 0
        self.num_relations = 0
        
        if mappings_path and os.path.exists(mappings_path):
            print(f"Loading mappings from {mappings_path}...")
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.num_entities = len(mappings.get('ent2id', []))
                self.num_relations = len(mappings.get('rel2id', []))
        else:
            print("Mapping file not found. Inferring counts from data...")
            self.num_entities = max(df[names[0]].max(), df[names[2]].max()) + 1
            self.num_relations = df[names[1]].max() + 1
            
        del df 
        print(f"Dataset loaded. Shape: {self.data.shape}")
        print(f"Entities: {self.num_entities}, Relations: {self.num_relations}")

    def __len__(self):
        """
        Returns the number of triples in the dataset.
        
        Returns:
            int: Number of triples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single triple by index.
        
        Args:
            idx (int): Index of the triple to retrieve.
        
        Returns:
            dict: Dictionary with keys 'head', 'relation', 'tail' containing
                  the corresponding entity/relation IDs as tensors.
        """
        return {
            'head': self.data[idx][0],
            'relation': self.data[idx][1],
            'tail': self.data[idx][2]
        }
