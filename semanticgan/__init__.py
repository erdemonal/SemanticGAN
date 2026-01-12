"""
SemanticGAN: A library for Knowledge Graph Completion using Wasserstein GANs.

This package provides models and utilities for training Wasserstein GANs on
knowledge graphs to generate candidate RDF triples.

Example:
    >>> from semanticgan import Generator, Discriminator, KnowledgeGraphDataset
    >>> dataset = KnowledgeGraphDataset("data/triples.txt")
    >>> G = Generator(embedding_dim=256, num_relations=dataset.num_relations)
    >>> D = Discriminator(num_entities=dataset.num_entities, num_relations=dataset.num_relations)
"""

from .models import Generator, Discriminator
from .dataset import KnowledgeGraphDataset

__version__ = "0.1.1"
__all__ = ["Generator", "Discriminator", "KnowledgeGraphDataset"]
