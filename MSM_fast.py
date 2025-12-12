import numpy as np
import math
import re
from datetime import datetime
from rapidfuzz import distance
from LSH import perform_LSH
from evaluation import *
from binary_vectors import *


def precompute_qgrams(strings, q=3, lowercase=True):

    qgrams = []
    pad = "#" * (q - 1)
    for s in strings:
        s = s.lower() if lowercase else s
        padded = pad + s + pad
        qset = {padded[i:i+q] for i in range(len(padded) - q + 1)}
        qgrams.append(qset)
    return qgrams

def fast_qgram_jaccard(set1, set2):

    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    qgram_similarity = intersection / union

    return qgram_similarity


def compute_hsm_similarity(model_words1, model_words2):
    
    set1, set2 = set(model_words1), set(model_words2)
    intersect = len(set1 & set2)
    union = len(set1 | set2)

    hsm_similarity = intersect / union if union != 0 else 0

    return hsm_similarity


def cosine_similarity(string1, string2):

    # Tokenize by splitting on whitespace
    words1 = set(string1.split())
    words2 = set(string2.split())

    numerator = len(words1.intersection(words2))
    denominator = math.sqrt(len(words1)) * math.sqrt(len(words2))

    similarity = numerator/denominator if denominator != 0 else 0

    return similarity

def lev_norm(string1, string2):
    # rapidfuzz returns similarity 0–100 → we want normalized distance 0–1
    if not string1 and not string2: return 0.0
    if not string1 or not string2: return 1.0

    sim = distance.Levenshtein.normalized_similarity(string1, string2)

    return 1.0 - sim

def separate_numeric(string):
    
    # Find all numeric parts (integer or decimal)
    numbers = re.findall(r'[-+]?\d*\.?\d+', string)

    # Remove numeric parts to get the non-numeric portion
    non_numeric = re.sub(r'[-+]?\d*\.?\d+', '', string)

    numeric_part = "".join(numbers)
    non_numeric_part = non_numeric.strip()

    return numeric_part, non_numeric_part

def avg_lv_sim(words1, words2):
    numerator = 0
    denominator = 0

    for w1 in words1:
        for w2 in words2:

            numerator += (1 - lev_norm(w1, w2)) * (len(w1) + len(w2))
            denominator += (len(w1) + len(w2))

    avg_lv_similarity = numerator / denominator if denominator != 0 else 0

    return avg_lv_similarity 

def avg_lv_sim_mw(model_words1, model_words2):
    numerator = 0
    denominator = 0

    for mw1 in model_words1:
        numeric_mw1, non_numeric_mw1 = separate_numeric(mw1)

        for mw2 in model_words2:
            numeric_mw2, non_numeric_mw2 = separate_numeric(mw2)

            lDistance_non_numeric = lev_norm(non_numeric_mw1, non_numeric_mw2)

            if lDistance_non_numeric < 0.5 and numeric_mw1 == numeric_mw2:

                numerator += (1 - lev_norm(mw1, mw2)) * (len(mw1) + len(mw2))
                denominator += (len(mw1) + len(mw2))

    avg_lv_similarity = numerator / denominator if denominator > 0 else 0.0

    return avg_lv_similarity

def compute_tmwm_similarity_fast(title1, title2, alpha, beta, delta=0.5):

    cosine_sim = cosine_similarity(title1, title2)
    if cosine_sim > alpha:
        return 1.0

    mw_title1 = extract_title_model_words(title1)
    mw_title2 = extract_title_model_words(title2)

    similar = False

    for mw1 in mw_title1:
        numeric_mw1, non_numeric_mw1 = separate_numeric(mw1)
        for mw2 in mw_title2:
            numeric_mw2, non_numeric_mw2 = separate_numeric(mw2)
            lDistance_non_numeric = lev_norm(non_numeric_mw1, non_numeric_mw2)

            if lDistance_non_numeric < 0.5:  # non-numeric parts are similar
                if numeric_mw1 != numeric_mw2:
                    return -1.0
                else:
                    similar = True

    # --- average Levenshtein over ALL title words (fast version) ---

    words1 = list(title1.split())
    words2 = list(title2.split())
    
    base_sim = beta * cosine_sim + (1-beta) * avg_lv_sim(words1, words2)

    if similar:
        return delta * avg_lv_sim_mw(mw_title1, mw_title2) + (1-delta) * base_sim 

    return base_sim 


def precompute_brands(data):
    
    brand_set = sorted({'acer', 'admiral', 'aiwa', 'akai', 'alba', 'amstrad', 'andrea smith electronics',
    'apex digital', 'apple', 'arcam', 'arise india', 'aga', 'audiovox', 'awa', 'baird',
    'bang & olufsen', 'beko', 'benq', 'binatone', 'blaupunkt', 'bpl', 'brionvega', 'bush',
    'canadian general electric', 'changhong', 'chimei', 'compal electronics', 'conar instruments',
    'continental edison', 'cossor', 'craig', 'crosley', 'curtis mathes', 'daewoo', 'dell',
    'delmonico', 'dumont', 'durabrand', 'dynatron', 'english electric', 'ekco', 'electrohome',
    'element electronics', 'emerson', 'emi', 'farnsworth', 'ferguson', 'ferranti', 'finlux',
    'fisher', 'fujitsu', 'funai', 'geloso', 'general electric', 'goodmans', 'google', 'gradiente',
    'graetz', 'grundig', 'haier', 'hallicrafters', 'hannspree', 'heath', 'hinari', 'hisense',
    'hitachi', 'hoffman', 'itel', 'jensen', 'jvc', 'kenmore', 'kent', 'kloss', 'kogan', 
    'kolster-brandes', 'konka', 'lanix', 'le.com', 'lg', 'loewe', 'luxor', 'magnavox', 'marantz',
    'marconiphone', 'matsui', 'memorex', 'micromax', 'metz', 'mitsubishi', 'mivar', 'motorola',
    'muntz', 'murphy', 'nec', 'nokia', 'nordmende', 'onida', 'orion', 'packard bell', 'panasonic',
    'pensonic', 'philco', 'philips', 'pioneer', 'planar', 'polaroid', 'proline', 'proscan', 'pye',
    'pyle', 'quasar', 'radioshack', 'rauland-borg', 'rca', 'realistic', 'rediffusion', 'saba',
    'salora', 'samsung', 'sansui', 'sanyo', 'schneider', 'seiki', 'seleco', 'setchell carlson',
    'sharp', 'siemens', 'skyworth', 'sony', 'soyo', 'stromberg-carlson', 'supersonic', 'sylvania',
    'symphonic', 'tandy', 'tatung', 'tcl', 'teleavia', 'telefunken', 'teletronics', 'thomson',
    'thorn', 'toshiba', 'tpv technology', 'tp vision', 'ultra', 'united states television', 
    'vestel', 'videocon', 'videoton', 'vizio', 'vu', 'walton', 'westinghouse', 'white-westinghouse',
    'xiaomi', 'zanussi', 'zenith', 'zonda'})

    # Precompile regex patterns for efficiency
    brand_patterns = [(b, re.compile(r'\b' + re.escape(b) + r'\b')) for b in brand_set]

    brands = []

    for row in data.itertuples():
        text = (row.title + " " + " ".join(str(v) for v in row.featuresMap.values())).lower()
        found = None

        # Find first brand that matches as a whole word
        for brand, pattern in brand_patterns:
            if pattern.search(text):
                found = brand
                break

        brands.append(found)

    return np.array(brands)


def perform_MSM(data, candidate_pairs, alpha, beta, gamma, mu, epsilon):

    def cluster_distance(cluster1, cluster2): 
        distances = [dissimilarity_matrix[i, j] for i in cluster1 for j in cluster2]

        # If ANY path between the clusters is ∞ → the cluster distance must be ∞
        if any(d == np.inf for d in distances):
            return np.inf

        # Otherwise use single-linkage (minimum finite distance)
        return min(distances)
    
    # ==================== PRECOMPUTATIONS ====================

    titles = data["title"].astype(str).tolist()
    shops = data["shop"].values
    features_maps = data["featuresMap"].tolist()

    # Brands
    brands = precompute_brands(data)

    # Key/Value q-grams (for KV component)
    all_keys = [list(fm.keys()) for fm in features_maps]
    all_values = [[str(v) for v in fm.values()] for fm in features_maps]

    key_qgrams = [precompute_qgrams(keys, q=3) for keys in all_keys]
    value_qgrams = [precompute_qgrams(vals, q=3) for vals in all_values]

    # Convert candidate pairs to numpy for fast filtering
    pairs = np.array(sorted(candidate_pairs), dtype=np.int32)

    if pairs.size == 0:
        print("    → 0 pairs provided. Returning singleton clusters.")
        return [{i} for i in range(len(data))]
    else:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    # ==================== EARLY FILTERING based on shop and brand ====================
    
    same_shop = shops[i_idx] == shops[j_idx]

    diff_brand = (
        (brands[i_idx] != None) &
        (brands[j_idx] != None) &
        (brands[i_idx] != brands[j_idx])
    )

    valid = ~(same_shop | diff_brand)

    pairs = pairs[valid]

    print(f"    → {len(pairs)} pairs remain after filtering")

    # ==================== DISSIMILARITY MATRIX ====================

    n = len(data)
    dissimilarity_matrix = np.full((n, n), np.inf)

    # dissimilarity_matrix = np.ones((n, n), dtype=float)
    # np.fill_diagonal(dissimilarity_matrix, 0.0)

    processed = 0
    for idx in range(len(pairs)):
        i, j = pairs[idx]

        if processed % 50000 == 0:
            print(f"       → {processed}/{len(pairs)} pairs processed [{datetime.now().strftime('%H:%M:%S')}]")
        processed += 1

        # --- Key-Value Component ---
        sim = 0
        kv_component = 0
        m = 0 # Number of matches
        w = 0 # Weight of matches

        ki_qgrams = key_qgrams[i]
        vi_qgrams = value_qgrams[i]
        kj_qgrams = key_qgrams[j]
        vj_qgrams = value_qgrams[j]

        fmi = features_maps[i]
        fmj = features_maps[j]

        keys_i = list(fmi.keys())
        keys_j = list(fmj.keys())

        unmatched_keys_i = set(keys_i)
        unmatched_keys_j = set(keys_j)

        for a, k1_set in enumerate(ki_qgrams):
            key_i = keys_i[a]
            for b, k2_set in enumerate(kj_qgrams):
                key_j = keys_j[b]

                keySim = fast_qgram_jaccard(k1_set, k2_set)
                if keySim > gamma:
                    valueSim = fast_qgram_jaccard(vi_qgrams[a], vj_qgrams[b])
                    weight = keySim
                    sim += weight * valueSim
                    m += 1
                    w += weight
                    
                    unmatched_keys_i.discard(key_i)
                    unmatched_keys_j.discard(key_j)

        if w > 0:
            kv_component = sim / w

        # --- HSM Component ---

        unmatched_fmi = {k: fmi[k] for k in unmatched_keys_i}
        unmatched_fmj = {k: fmj[k] for k in unmatched_keys_j}

        mwi = extract_features_model_words(unmatched_fmi)
        mwj = extract_features_model_words(unmatched_fmj)

        hsm_component = compute_hsm_similarity(mwi, mwj)

        # --- TMWM Component ---

        title_i = titles[i]
        title_j = titles[j]

        tmwm_component = compute_tmwm_similarity_fast(title_i, title_j, alpha, beta)

        # --- Final MSM Score ---

        if tmwm_component == -1:  
            theta1 = m / (min(len(features_maps[i]), len(features_maps[j])) + 1e-8)
            msm_similarity = theta1 * kv_component + (1 - theta1) * hsm_component
        else:
            theta1 = (1 - mu) * (m / (min(len(features_maps[i]), len(features_maps[j])) + 1e-8))
            theta2 = 1 - mu - theta1
            msm_similarity = theta1 * kv_component + theta2 * hsm_component + mu * tmwm_component

        dissimilarity_matrix[i, j] = 1 - msm_similarity
        dissimilarity_matrix[j, i] = 1 - msm_similarity

    # ==================== SINGLE-LINKAGE CLUSTERING ====================

    clusters = [{i} for i in range(n)] # Start with each product alone
    threshold = epsilon

    while True:
        best_pair = None
        best_dist = np.inf
        # best_dist = 1.0

        # Find closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance(clusters[i], clusters[j])

                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        # Stop if smallest inter-cluster distance exceeds threshold
        if best_dist > threshold:
            break

        # Otherwise merge the best clusters
        i, j = best_pair
        new_cluster = clusters[i] | clusters[j]

        # Remove old clusters and insert the merged one
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_cluster)

    return clusters


def make_objective(indices, data, minhash_signatures, true_pairs, rows, bands):

    # Perform LSH
    print(f"-> performing LSH... [{datetime.now().strftime('%H:%M:%S')}]")
    cached_candidate_pairs = perform_LSH(minhash_signatures, rows, bands)
    print(f"    -> Succesfully found {len(cached_candidate_pairs)} candidate pairs [{datetime.now().strftime('%H:%M:%S')}]")

    def objective(trial):

        # MSM parameters to tune
        epsilon = trial.suggest_float("epsilon", 0.0, 1.0, step=0.1)

        print(f"epsilon={epsilon:.3f}")

        # LSH
        candidate_pairs = cached_candidate_pairs
        
        # Perform MSM
        print(f"-> performing MSM... [{datetime.now().strftime('%H:%M:%S')}]")
        clusters = perform_MSM(
            data, 
            candidate_pairs,
            alpha=0.602, 
            beta=0.000, 
            gamma=0.756, 
            mu=0.650, 
            epsilon=epsilon
        )
        print(f"    -> MSM completed [{datetime.now().strftime('%H:%M:%S')}]")

        # Convert cluster indices to global indices
        cluster_pairs = []
        for cluster in clusters:
            items = list(cluster)
            if len(items) > 1:
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair_train_indices = tuple(
                            sorted((indices[items[i]], indices[items[j]]))
                        )
                        cluster_pairs.append(pair_train_indices)

        # Compute metrics
        total_duplicates = len(true_pairs)
        num_comparisons_msm = len(candidate_pairs)
        duplicates_found_msm = len(set(true_pairs) & set(cluster_pairs))
        
        pq_msm = compute_pair_quality(duplicates_found_msm, num_comparisons_msm)
        pc_msm = compute_pair_completeness(duplicates_found_msm, total_duplicates)
        f1 = compute_F1(pq_msm, pc_msm)

        # Optuna maximizes objective by default, so return f1
        return f1

    return objective

    
 