import random
from sympy import nextprime
import numpy as np

def build_minhash_signatures(binary_vectors, n_permutations, seed):

    random.seed(seed)
    num_words, num_products = binary_vectors.shape
    signature_matrix = np.full((n_permutations, num_products), np.inf)

    # --- Generate random prime number ---

    base = n_permutations + random.randint(1, 5000) # Range of 5K ensures sufficient randomness
    p = nextprime(base)

    # --- Generate MinHash permutations (a, b) ---

    permutations = [
        (random.randint(1, p-1), random.randint(1, p-1)) # [1,p-1] avoids degenerate hashes, all values above p collapse back down
        for _ in range(n_permutations)
    ]

    # --- Apply MinHash ---

    for word_idx in range(num_words): # for each row r

        hashed_vals = [
            (a + b * word_idx) % p
            for (a, b) in permutations
        ]  # for each function hi compute hi(r)

        for prod_idx in range(num_products): # for each column c
            if binary_vectors[word_idx, prod_idx] == 1: # if c has 1 in row r
                signature_matrix[:, prod_idx] = np.minimum(
                    signature_matrix[:, prod_idx],
                    hashed_vals
                    )  # if hi(r) < M(i, c), then M(i, c) = hi(r)

    return signature_matrix