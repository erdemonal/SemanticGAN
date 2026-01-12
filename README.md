# Wasserstein GAN for Knowledge Graph Completion

[![PyPI version](https://badge.fury.io/py/semantic-gan.svg)](https://badge.fury.io/py/semantic-gan)
[![Sync Results](https://github.com/erdemonal/SemanticGAN/actions/workflows/sync-results.yml/badge.svg)](https://github.com/erdemonal/SemanticGAN/actions/workflows/sync-results.yml)
[![pages-build-deployment](https://github.com/erdemonal/SemanticGAN/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/erdemonal/SemanticGAN/actions/workflows/pages/pages-build-deployment)

`semantic-gan` is a Python implementation of a Wasserstein GAN architecture for knowledge graph completion.

## Installation

The package can be installed from PyPI:

```bash
pip install semantic-gan
```

Or install from source:

```bash
git clone https://github.com/erdemonal/SemanticGAN.git
cd SemanticGAN
pip install -e .
```

## Usage

The following example demonstrates usage with a generic knowledge graph dataset:

```python
from semanticgan import KnowledgeGraphDataset, Generator, Discriminator
import torch
from torch.utils.data import DataLoader

# 1. Load a generic knowledge graph dataset
# Format: head_id [tab] relation_id [tab] tail_id
dataset = KnowledgeGraphDataset(
    triples_path="my_custom_data.txt", 
    sep='\t', 
    names=['h', 'r', 't']
)

# 2. Initialize Models
G = Generator(
    embedding_dim=256, 
    hidden_dim=512, 
    num_relations=dataset.num_relations
)
D = Discriminator(
    num_entities=dataset.num_entities,
    num_relations=dataset.num_relations,
    embedding_dim=256,
    hidden_dim=512
)

# 3. Create data loader and train
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
```

## Technical Report: DBLP Case Study

This repository accompanies a technical report entitled
"Knowledge Graph Completion and RDF Triple Generation with a Wasserstein GAN",
presenting an experimental study on the DBLP Computer Science Bibliography.

### Technical Report

A detailed description of the model architecture, training procedure, and evaluation protocol is provided in the technical report:

[`paper/knowledge-graph-completion-wasserstein-gan.pdf`](https://github.com/erdemonal/SemanticGAN/blob/main/paper/knowledge-graph-completion-wasserstein-gan.pdf)

The LaTeX source is available in [`paper/main.tex`](https://github.com/erdemonal/SemanticGAN/blob/main/paper/main.tex)

### Results

Training artifacts and generated RDF triples are available at:
https://erdemonal.github.io/SemanticGAN

### Methodology

The preprocessing pipeline parses the DBLP XML dump from https://dblp.uni-trier.de/xml to extract a knowledge graph with entity types Publication, Author, Venue, and Year. Relations include dblp:wrote, dblp:hasAuthor, dblp:publishedIn, and dblp:inYear.

The preprocessing script `scripts/prepare_dblp_kg.py` reads the XML file incrementally and produces RDF triples in tab separated format. The preprocessed 1M triple dataset is versioned and maintained in the [Hugging Face Dataset Hub](https://huggingface.co/datasets/erdemonal/SemanticGAN-Dataset).

The WGAN model consists of a Generator that produces tail entity embeddings from noise and relation embeddings, and a Discriminator that scores triples using a scalar Wasserstein distance. Training uses RMSprop with gradient clipping to enforce the Lipschitz constraint.

Training and synchronization are automated via a continuous integration workflow. Training is executed on external compute infrastructure, and the resulting artifacts are synchronized after each run.

### Model Storage and Data Decoupling

Model weights and processed knowledge graph artifacts are hosted on the Hugging Face Hub across two repositories:

**Model Hub:** [erdemonal/SemanticGAN](https://huggingface.co/erdemonal/SemanticGAN) stores the persistent WGAN checkpoints.

**Dataset Hub:** [erdemonal/SemanticGAN-Dataset](https://huggingface.co/datasets/erdemonal/SemanticGAN-Dataset) contains the processed DBLP triples and ID mappings.

The automated training workflow fetches processed data from the Dataset Hub and restores model states from the Model Hub before each training run.

### Data Availability

The DBLP dataset is publicly available from https://dblp.uni-trier.de/xml

Documentation is available at https://dblp.org/xml/docu/dblpxml.pdf
