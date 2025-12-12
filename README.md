# Scalable-Product-Duplication-Detection-using-LSH

This repository contains the implementation and evaluation code for the paper Scalable Product Duplicate Detection using LSH. The work extends the state-of-the-art MSMP+ framework for scalable duplicate detection in product data aggregation from webshops. Key contributions include:

• A global data-cleaning procedure to remove non-alphanumeric tokens (with exceptions for – and /).

• A unified model word definition for product titles and feature values, combined with frequency-based filtering of features (retaining only those in ≥10% of products).

• These extensions (termed MSMP++) reduce noise in binary vector representations for Locality-Sensitive Hashing (LSH), improving candidate pair generation for downstream clustering.

The approach is evaluated on a dataset of 1,624 televisions from four webshops (Amazon, Best Buy, The Nerds, Newegg), achieving a 12% increase in F1* score over MSMP+.

## Methodology Overview
### Data Cleaning

Replace all non-alphanumeric tokens (except – and /) with spaces; remove – and / entirely.
Prevents mismatches like (120hz vs. 120hz or 18–3/8inch vs. 1838inch.

### Binary Vector Representations

Extract model words from cleaned titles and features using a unified definition: combinations of numeric and alphabetic/punctuation tokens (e.g., 28inch from both titles and features).
Filter features by frequency (≥10% of products) to exclude noise (e.g., Estimated Operation Cost).
Construct binary vectors where each dimension represents a unique model word.

### MinHash Signatures

Use single-pass MinHash with k permutations (set to 50% of binary vector rows) and hash functions h_{a,b}(x) = (a + b x) mod p.
Reduces vectors to compact signatures for efficient Jaccard similarity estimation.

### Locality-Sensitive Hashing (LSH)

Divide signatures into b bands of r rows each (k = r * b).
Hash band signatures into buckets using h(s) = (s · a) mod p; pairs in the same bucket (in ≥1 band) are candidates.

### Evaluation Metrics

Pair Completeness (PC): Fraction of true duplicates in candidate pairs.
Pair Quality (PQ): Fraction of candidate pairs that are true duplicates.
F1*/F1: Harmonic mean of PC and PQ, balanced for scalability.

## Installation
This project uses Python 3.8+ and requires the following dependencies:
'pip install numpy scipy pandas scikit-learn optuna matplotlib seaborn'

## Usage
1. Download the full content of this repository
2. Run MSMP+ for baseline results (outputs MSMP+ graphs)
3. Run MSMP++ for extension results (outputs MSMP++ with MSMP+ baseline)

Note: code is written such that runtime can be terminated and restarted at any time, without losing progress (due to long runtime for MSM).
