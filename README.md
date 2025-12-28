# Wasserstein GAN for Knowledge Graph Completion

[![Sync Results](https://github.com/erdemonal/SemanticGAN/actions/workflows/sync-results.yml/badge.svg)](https://github.com/erdemonal/SemanticGAN/actions/workflows/sync-results.yml)
[![pages-build-deployment](https://github.com/erdemonal/SemanticGAN/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/erdemonal/SemanticGAN/actions/workflows/pages/pages-build-deployment)

This repository contains a research prototype for Knowledge Graph Completion on the DBLP Computer Science Bibliography.

The system uses a Wasserstein GAN to generate candidate RDF triples from an evolving publication graph. Model training is executed periodically using an automated workflow.

## Technical Report

A detailed description of the model, training procedure, and evaluation is provided in the technical report:

[`paper/WGAN_Knowledge_Graph_Completion.pdf`](paper/WGAN_Knowledge_Graph_Completion.pdf)

The LaTeX source is available in [`paper/main.tex`](paper/main.tex)

## Results

Training outputs and generated RDF triples are available at:
https://erdemonal.github.io/SemanticGAN

## Methodology

The system processes the DBLP XML dump from https://dblp.uni-trier.de/xml to extract a knowledge graph with entity types Publication, Author, Venue, and Year. Relations include dblp:wrote, dblp:hasAuthor, dblp:publishedIn, and dblp:inYear.

The preprocessing script `scripts/prepare_dblp_kg.py` reads the XML file incrementally and produces RDF triples in tab separated format. The preprocessed 1M triple dataset is versioned and maintained in the [Hugging Face Dataset Hub](https://huggingface.co/datasets/erdemonal/SemanticGAN-Dataset).

The WGAN model consists of a Generator that produces tail entity embeddings from noise and relation embeddings, and a Discriminator that scores triples using a scalar Wasserstein distance. Training uses RMSprop with gradient clipping to enforce the Lipschitz constraint.

An automated training workflow is orchestrated via GitHub Actions. Training is executed on external compute infrastructure, and the resulting artifacts are synchronized after each run.

## Model Storage and Data Decoupling

Model weights and processed knowledge graph artifacts are hosted on the Hugging Face Hub across two repositories:

Model Hub: [erdemonal/SemanticGAN](https://huggingface.co/erdemonal/SemanticGAN) stores the persistent WGAN checkpoints.

Dataset Hub: [erdemonal/SemanticGAN-Dataset](https://huggingface.co/erdemonal/SemanticGAN-Dataset) contains the processed DBLP triples and ID mappings.

The automated training workflow fetches processed data from the Dataset Hub and restores model states from the Model Hub before each training run.

## Repository Structure

Technical report in `paper/`, preprocessing scripts in `scripts/`, model code in `src/`, data in `data/`, checkpoints in `checkpoints/`.

## Data Availability

The DBLP dataset is publicly available from https://dblp.uni-trier.de/xml

Documentation is available at https://dblp.org/xml/docu/dblpxml.pdf

