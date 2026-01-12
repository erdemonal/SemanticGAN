"""
Wasserstein GAN models for Knowledge Graph Completion.

This module provides Generator and Discriminator models for training
a Wasserstein GAN on knowledge graph triples.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for Wasserstein GAN.
    
    Generates tail entity embeddings from random noise and relation embeddings.
    Used in the WGAN framework to produce candidate knowledge graph triples.
    
    Args:
        embedding_dim (int): Dimension of entity and relation embeddings. Default: 256.
        hidden_dim (int): Dimension of hidden layers. Default: 512.
        num_relations (int): Number of unique relations in the knowledge graph. Required.
    
    Example:
        >>> G = Generator(embedding_dim=256, hidden_dim=512, num_relations=35)
        >>> noise = torch.randn(32, 256)  # batch_size=32
        >>> relation_ids = torch.randint(0, 35, (32,))
        >>> tail_embeddings = G(noise, relation_ids)
        >>> print(tail_embeddings.shape)  # torch.Size([32, 256])
    """
    
    def __init__(self, embedding_dim=256, hidden_dim=512, num_relations=0):
        super(Generator, self).__init__()
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, noise, relation_ids):
        """
        Forward pass of the generator.
        
        Args:
            noise (torch.Tensor): Random noise tensor of shape (batch_size, embedding_dim).
            relation_ids (torch.Tensor): Relation IDs tensor of shape (batch_size,).
        
        Returns:
            torch.Tensor: Generated tail entity embeddings of shape (batch_size, embedding_dim).
        """
        r_emb = self.rel_embedding(relation_ids)
        x = torch.cat([noise, r_emb], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    """
    Discriminator network for Wasserstein GAN.
    
    Scores knowledge graph triples using Wasserstein distance. Takes head entity IDs,
    relation IDs, and tail entity embeddings to produce a scalar score indicating
    the quality/realness of the triple.
    
    Args:
        num_entities (int): Number of unique entities in the knowledge graph. Required.
        num_relations (int): Number of unique relations in the knowledge graph. Required.
        embedding_dim (int): Dimension of entity and relation embeddings. Default: 256.
        hidden_dim (int): Dimension of hidden layers. Default: 512.
    
    Example:
        >>> D = Discriminator(num_entities=10000, num_relations=35, embedding_dim=256)
        >>> head_ids = torch.randint(0, 10000, (32,))
        >>> rel_ids = torch.randint(0, 35, (32,))
        >>> tail_emb = torch.randn(32, 256)
        >>> scores = D(head_ids, rel_ids, tail_emb)
        >>> print(scores.shape)  # torch.Size([32, 1])
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim=256, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.ent_embedding = nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, head_ids, rel_ids, tail_embedding):
        """
        Forward pass of the discriminator.
        
        Args:
            head_ids (torch.Tensor): Head entity IDs of shape (batch_size,).
            rel_ids (torch.Tensor): Relation IDs of shape (batch_size,).
            tail_embedding (torch.Tensor): Tail entity embeddings of shape (batch_size, embedding_dim).
        
        Returns:
            torch.Tensor: Wasserstein scores of shape (batch_size, 1). Higher scores indicate more realistic triples.
        """
        h_emb = self.ent_embedding(head_ids)
        r_emb = self.rel_embedding(rel_ids)
        x = torch.cat([h_emb, r_emb, tail_embedding], dim=1)
        return self.net(x)

    def get_entity_embedding(self, entity_ids):
        """
        Get entity embeddings for given entity IDs.
        
        Useful for retrieving embeddings of existing entities or for nearest neighbor search
        when generating new triples.
        
        Args:
            entity_ids (torch.Tensor): Entity IDs of shape (batch_size,).
        
        Returns:
            torch.Tensor: Entity embeddings of shape (batch_size, embedding_dim).
        """
        return self.ent_embedding(entity_ids)