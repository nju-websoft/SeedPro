# SeedPro: Semantic-Enhanced Dataset Profiling for Search and Recommendation

This repository contains the code implementation for SeedPro, a semantics-enhanced approach to dataset profiling aimed at improving dataset search and recommendation tasks. Below are the instructions for setting up and running the code.

## Environment

Python version: 3.8
Partial requirements: scipy, ranx, pymysql, sklearn, nltk, text2vec, refined

## Directory Structure

```
SeedPro
├─ data
│  ├─ BDR_metadata
│  ├─ bm25
│  ├─ distributed
│  ├─ ontological
│  ├─ qrels
│  └─ results
├─ README.md
└─ src
   ├─ common.py
   ├─ constants.py
   ├─ database.py
   ├─ entity_retain.py
   ├─ logging.conf
   ├─ ner.py
   ├─ reproduce_keds
   │  ├─ collect_res.py
   │  ├─ configs.py
   │  ├─ database.py
   │  ├─ eval.py
   │  ├─ kgtk_similarity.ipynb
   │  ├─ kgtk_similarity_for_BDR.ipynb
   │  ├─ logger.py
   │  ├─ merge_score.py
   │  ├─ pooling.ipynb
   │  ├─ reformat_sparse_res.py
   │  ├─ retrieve_all.py
   │  └─ util.py
   ├─ retrieve.py
   ├─ scripts.py
   ├─ search_recommend.py
   ├─ tfidf.py
   └─ transitive_closure.py

```

## Data

data directory contains the following subdirectories:
- bm25: Baseline results
- qrels: Qrels for test collections
- ontological, distributed: Results using different semantic approaches
- results: Results of re-ranking
- BDR_matdata: Data for the BDR test collection in CSV format

Results in text format follow the structure: query_id, dataset_id, score (tab-separated).

## Source Code

src directory contains:
- reproduce_keds: Code for reproducing Luo et al.'s paper on dataset profiling with distributed semantics
- ner.py: Entity recognition and linking code
- transitive_closure.py: Transitive closure computation code
- tfidf.py: Frequency-based weighting code
- search_recommend.py: Code for search and recommendation experiments
- scripts: Code for displaying paper results

Other basic code files are not listed.

Note: The data of BDR exceeds the limit of Github, please access via https://doi.org/10.5281/zenodo.10991770