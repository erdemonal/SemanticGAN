# Wasserstein GAN for Knowledge Graph Completion with Continuous Learning

[![Daily Semantic GAN Training](https://github.com/erdemonal11/schema-guided-rdf-generation/actions/workflows/daily-experiment.yml/badge.svg)](https://github.com/erdemonal11/schema-guided-rdf-generation/actions/workflows/daily-experiment.yml)
[![pages-build-deployment](https://github.com/erdemonal11/schema-guided-rdf-generation/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/erdemonal11/schema-guided-rdf-generation/actions/workflows/pages/pages-build-deployment)

A Wasserstein Generative Adversarial Network (WGAN) for Knowledge Graph Completion on the DBLP Computer Science Bibliography dataset. Generates RDF triples representing scientific publication relationships using a continuous learning pipeline that retrains the model daily via automated CI/CD workflows.

## Technical Research Report

Access the technical report: [`paper/WGAN_Knowledge_Graph_Completion.pdf`](paper/WGAN_Knowledge_Graph_Completion.pdf)

The report is also available in LaTeX source: [`paper/main.tex`](paper/main.tex)

## Live Dashboard

Live metrics are available at [https://erdemonal11.github.io/schema-guided-rdf-generation](https://erdemonal11.github.io/schema-guided-rdf-generation/)

The dashboard displays novelty scores (percentage of generated triples not in training data), diversity scores (distinct relation types), generated RDF hypotheses with confidence scores, and training loss convergence curves.

## Methodology

The system processes the DBLP XML dump from [dblp.uni-trier.de/xml](https://dblp.uni-trier.de/xml/) to extract a knowledge graph with entity types Publication, Author, Venue, and Year. Relations include dblp:wrote (Author to Publication), dblp:hasAuthor (Publication to Author), dblp:publishedIn (Publication to Venue), and dblp:inYear (Publication to Year).

The preprocessing script `scripts/prepare_dblp_kg.py` streams the XML file, extracts publication metadata, and produces RDF triples in tab-separated format.

The WGAN model consists of a Generator that takes random noise and a relation embedding to produce a tail entity embedding, and a Discriminator that scores triples for plausibility using entity and relation embeddings. The Discriminator outputs a scalar Wasserstein distance rather than a probability. Training uses RMSprop optimization with gradient clipping (clamp value 0.01) to enforce the Lipschitz constraint required for Wasserstein distance.

The continuous learning pipeline runs via GitHub Actions in `.github/workflows/daily-experiment.yml`. The workflow runs daily at 02:00 UTC, loads the latest checkpoint for incremental training, computes novelty and diversity metrics on generated triples, updates the dashboard, and commits model checkpoints and metrics to the repository.

## Repository Structure

Technical report in `paper/`, preprocessing scripts in `scripts/`, model code in `src/`, data in `data/`, checkpoints in `checkpoints/`, dashboard in `index.html`.

## Experimental Results

Empirical evaluation on the DBLP dataset confirms stable Wasserstein loss convergence without mode collapse. Generated triples include valid author-venue pairings not present in training data. The model generates across all relation types (wrote, hasAuthor, publishedIn, inYear) rather than collapsing to a single relation. Detailed metrics are available on the live dashboard.

## Related Work

This work builds upon KBGAN (Cai & Wang, 2018) for adversarial negative sampling in Knowledge Graph Embeddings, WGAN for Graphs (Dai et al., 2020) on Wasserstein distance for discrete graph data, and Continual Learning (Daruna et al., 2021) for adaptation to evolving knowledge graphs. Full references are provided in the technical report.

## Data Availability

The DBLP dataset is publicly available from [dblp.uni-trier.de/xml/](https://dblp.uni-trier.de/xml/). Documentation is available at [DBLP — Some Lessons Learned](https://dblp.org/xml/docu/dblpxml.pdf). Place the `dblp.xml` file in `data/real/` before running preprocessing.

**Note**: The continuous learning pipeline is active and updates the model daily via automated workflows.
