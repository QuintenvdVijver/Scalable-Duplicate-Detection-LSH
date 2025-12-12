# Scalable-Product-Duplication-Detection-using-LSH

This repository contains the implementation and evaluation code for the paper Scalable Product Duplicate Detection using LSH. The work extends the state-of-the-art MSMP+ framework for scalable duplicate detection in product data aggregation from webshops. Key contributions include:

• A global data-cleaning procedure to remove non-alphanumeric tokens (with exceptions for – and /).
• A unified model word definition for product titles and feature values, combined with frequency-based filtering of features (retaining only those in ≥10% of products).
• These extensions (termed MSMP++) reduce noise in binary vector representations for Locality-Sensitive Hashing (LSH), improving candidate pair generation for downstream clustering.

The approach is evaluated on a dataset of 1,624 televisions from four webshops (Amazon, Best Buy, The Nerds, Newegg), achieving a 12% increase in F1* score over MSMP+.
