# Wasserstein GAN for Knowledge Graph Completion with Continuous Learning

[![Daily Semantic GAN Training](https://github.com/erdemonal11/SemanticGAN/actions/workflows/daily-experiment.yml/badge.svg)](https://github.com/erdemonal11/SemanticGAN/actions/workflows/daily-experiment.yml)
[![pages-build-deployment](https://github.com/erdemonal11/SemanticGAN/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/erdemonal11/SemanticGAN/actions/workflows/pages/pages-build-deployment)

This repository contains a research prototype for Knowledge Graph Completion on the DBLP Computer Science Bibliography.

The system uses a Wasserstein GAN to generate candidate RDF triples from an evolving publication graph. Model training is updated daily using an automated workflow.

## Technical Report

A detailed description of the model, training procedure, and evaluation is provided in the technical report:

[`paper/WGAN_Knowledge_Graph_Completion.pdf`](paper/WGAN_Knowledge_Graph_Completion.pdf)

The LaTeX source is available in [`paper/main.tex`](paper/main.tex).

## Training Dashboard

Metrics are available at  
https://erdemonal11.github.io/SemanticGAN

The dashboard is used to monitor training progress and inspect generated RDF triples.

## Methodology

The system processes the DBLP XML dump from https://dblp.uni-trier.de/xml to extract a knowledge graph with entity types Publication, Author, Venue, and Year. Relations include dblp:wrote, dblp:hasAuthor, dblp:publishedIn, and dblp:inYear.

The preprocessing script `scripts/prepare_dblp_kg.py` streams the XML file and produces RDF triples in tab separated format.

The WGAN model consists of a Generator that produces tail entity embeddings from noise and relation embeddings, and a Discriminator that scores triples using a scalar Wasserstein distance. Training uses RMSprop with gradient clipping to enforce the Lipschitz constraint.

The continuous learning pipeline runs via GitHub Actions in `.github/workflows/daily-experiment.yml`. The workflow loads the latest checkpoint, updates the model with new data when available, computes evaluation metrics, and updates the dashboard.

## Repository Structure

Technical report in `paper/`, preprocessing scripts in `scripts/`, model code in `src/`, data in `data/`, checkpoints in `checkpoints/`, dashboard files in `index.html`.

## Experimental Results

Experiments on the DBLP dataset show stable Wasserstein loss behavior during training. Generated triples span all relation types and include plausible author venue associations not present in the training set. Detailed metrics are available on the live dashboard.

## Data Availability

The DBLP dataset is publicly available from https://dblp.uni-trier.de/xml. Place the `dblp.xml` file in `data/real/` before running preprocessing.

Note: The continuous learning pipeline is active and updates the model daily.
