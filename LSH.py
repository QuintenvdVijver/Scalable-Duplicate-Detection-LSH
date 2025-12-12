from collections import defaultdict

def perform_LSH(sig_matrix, rows, bands):
    r = rows
    b = bands
    num_products = sig_matrix.shape[1]

    # Choose a large prime for modulo hashing
    p = 2**61 - 1
    a = 31  # small constant for polynomial hash

    buckets = [defaultdict(list) for _ in range(b)]
    candidate_pairs = set()

    # --- Explicit hash for each band ---
    def hash_band(band_signature):
        h = 0
        for idx, val in enumerate(band_signature):
            h = (h * a + val) % p
        return h

    # --- Hash each band for each product ---
    for i in range(num_products):  # For each column
        for band in range(b):
            start = band * r
            end = min((band + 1) * r, sig_matrix.shape[0])
            band_signature = sig_matrix[start:end, i]
            band_hash = hash_band(band_signature)
            buckets[band][band_hash].append(i)

    # --- Collect candidate pairs ---
    for band_bucket in buckets:
        for bucket_products in band_bucket.values():
            if len(bucket_products) > 1:
                for j in range(len(bucket_products)):
                    for k in range(j + 1, len(bucket_products)):
                        candidate_pairs.add((bucket_products[j], bucket_products[k]))

    return candidate_pairs