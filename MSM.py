import numpy as np
from binary_vectors import extract_title_model_words, extract_features_model_words
import math
import re
import Levenshtein
from datetime import datetime
from LSH import perform_LSH
from evaluation import *


def compute_qgram_similarity(string1, string2, q):

    # Boundary padding with (q−1) dummy characters on both sides
    pad = "#" * (q - 1)
    s1 = pad + string1 + pad
    s2 = pad + string2 + pad

    q_grams_1 = {s1[i:i+q] for i in range(len(s1) - q + 1)}
    q_grams_2 = {s2[i:i+q] for i in range(len(s2) - q + 1)}

    # Compute q-gram distance: number of different occurrences
    qgram_distance = len(q_grams_1.symmetric_difference(q_grams_2))

    n1 = len(q_grams_1)
    n2 = len(q_grams_2)

    qgram_similarity = (n1 + n2 - qgram_distance) / (n1 + n2) if n1 + n2 != 0 else 0

    return qgram_similarity

def compute_hsm_similarity(model_words1, model_words2):
    
    set1, set2 = set(model_words1), set(model_words2)
    intersect = len(set1 & set2)
    union = len(set1 | set2)

    hsm_similarity = intersect / union if union != 0 else 0

    return hsm_similarity

def compute_tmwm_similarity(title1, title2, alpha, beta, delta):

    def cosine_similarity(string1, string2):

        # Tokenize by splitting on whitespace
        words1 = set(string1.split())
        words2 = set(string2.split())

        numerator = len(words1.intersection(words2))
        denominator = math.sqrt(len(words1)) * math.sqrt(len(words2))

        similarity = numerator/denominator if denominator != 0 else 0

        return similarity
    
    def seperate_numeric(string):
        
        # Find all numeric parts (integer or decimal)
        numbers = re.findall(r'[-+]?\d*\.?\d+', string)

        # Remove numeric parts to get the non-numeric portion
        non_numeric = re.sub(r'[-+]?\d*\.?\d+', '', string)

        numeric_part = "".join(numbers)
        non_numeric_part = non_numeric.strip()

        return numeric_part, non_numeric_part

    def Levenshtein_norm(string1, string2):
        lv_distance = Levenshtein.distance(string1, string2)
        max_string_length = max(len(string1), len(string2))

        lv_distance_norm = lv_distance/max_string_length if max_string_length != 0 else 0

        return lv_distance_norm

    def avg_lv_sim(words1, words2):
        numerator = 0
        denominator = 0

        for w1 in words1:
            for w2 in words2:

                numerator += (1 - Levenshtein_norm(w1, w2)) * (len(w1) + len(w2))
                denominator += (len(w1) + len(w2))

        avg_lv_similarity = numerator / denominator if denominator != 0 else 0

        return avg_lv_similarity 
    
    def avg_lv_sim_mw(model_words1, model_words2):
        numerator = 0
        denominator = 0

        for mw1 in model_words1:
            numeric_mw1, non_numeric_mw1 = seperate_numeric(mw1)

            for mw2 in model_words2:
                numeric_mw2, non_numeric_mw2 = seperate_numeric(mw2)

                lDistance_non_numeric = Levenshtein_norm(non_numeric_mw1, non_numeric_mw2)

                if lDistance_non_numeric < 0.5 and numeric_mw1 == numeric_mw2:

                    numerator += (1 - Levenshtein_norm(mw1, mw2)) * (len(mw1) + len(mw2))
                    denominator += (len(mw1) + len(mw2))

        avg_lv_similarity = numerator / denominator if denominator != 0 else 0

        return avg_lv_similarity 
    
    cosine_sim = cosine_similarity(title1, title2)

    if cosine_sim > alpha:
        return 1
    
    mw_title1 = extract_title_model_words(title1)
    mw_title2 = extract_title_model_words(title2)

    similar = False

    # Iterate over all model words
    for mw1 in mw_title1:
        numeric_mw1, non_numeric_mw1 = seperate_numeric(mw1)

        for mw2 in mw_title2:
            numeric_mw2, non_numeric_mw2 = seperate_numeric(mw2)

            lDistance_numeric = Levenshtein_norm(numeric_mw1, numeric_mw2) if numeric_mw1 and numeric_mw2 else 1
            lDistance_non_numeric = Levenshtein_norm(non_numeric_mw1, non_numeric_mw2)


            if lDistance_non_numeric < 0.5 and numeric_mw1 != numeric_mw2: 
                return -1
            elif lDistance_non_numeric < 0.5 and numeric_mw1 == numeric_mw2:
                similar = True

    words1 = set(title1.split())
    words2 = set(title2.split())

    base_sim = beta * cosine_sim + (1-beta) * avg_lv_sim(words1, words2)

    if similar:
        return delta * avg_lv_sim_mw(mw_title1, mw_title2) + (1-delta) * base_sim 

    return base_sim 

def perform_MSM(data, candidate_pairs, alpha, beta, gamma, mu, epsilon):

    def cluster_distance(cluster1, cluster2): 
        distances = [dissimilarity_matrix[i, j] for i in cluster1 for j in cluster2]

        # If ANY path between the clusters is ∞ → the cluster distance must be ∞
        if any(d == np.inf for d in distances):
            return np.inf

        # Otherwise use single-linkage (minimum finite distance)
        return min(distances)

    def extract_brand(product):

        brands = [
        'acer', 'admiral', 'aiwa', 'akai', 'alba', 'amstrad', 'andrea smith electronics',
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
        'xiaomi', 'zanussi', 'zenith', 'zonda'
        ]
        
        # 'insignia', 'coby', 'naxa', 'viewsonic', 'sunbritetv', 'optoma', 'venturer', 'dynex', 

        title_text = product["title"]
        features_text = " ".join([str(v) for v in product["featuresMap"].values()])
        text = title_text + " " + features_text

        # Check brands
        for brand in brands:
            if brand in text:
                return brand  # first match
            
        return None

    # Initialize dissimilarity matrix
    n = len(data) # Number of products
    dissimilarity_matrix = np.full((n, n), np.inf) # Sets non-candidate pairs automatically to inf
    
    # Compute dissimilarities only for candidate pairs
    print(f"    -> evaluating candidate pairs... [{datetime.now().strftime('%H:%M:%S')}]")
    candidate_pairs_compared = 0
    for i, j in candidate_pairs:

        candidate_pairs_compared += 1

        if candidate_pairs_compared % 100000 == 0:
            print(f"       -> {candidate_pairs_compared} compared... [{datetime.now().strftime('%H:%M:%S')}]")

        product_i = data.iloc[i]
        product_j = data.iloc[j]
        title_i = product_i["title"]
        title_j = product_j["title"]
        shop_i = product_i["shop"]
        shop_j = product_j["shop"]
        brand_i = extract_brand(product_i) 
        brand_j = extract_brand(product_j)

        # Leave dissimilarity as infinity for same shops or different brands
        if shop_i == shop_j:
            continue

        if brand_i is not None and brand_j is not None and brand_i != brand_j:
            continue

        # Else compute msm similarity
        else:
            sim = 0
            kv_component = 0
            m = 0 # Number of matches
            w = 0 # Weight of matches

            fmi = product_i["featuresMap"]
            fmj = product_j["featuresMap"]

            kvp_i = list(fmi.keys())
            kvp_j = list(fmj.keys())

            unmatched_i = set(kvp_i)
            unmatched_j = set(kvp_j)

            for key_i in kvp_i:
                for key_j in kvp_j:
                    keySim = compute_qgram_similarity(key_i, key_j,3)
                    if keySim > gamma:
                        valueSim = compute_qgram_similarity(fmi[key_i], fmj[key_j],3)
                        weight = keySim
                        sim += weight * valueSim
                        m += 1
                        w += weight
                        unmatched_i.discard(key_i)
                        unmatched_j.discard(key_j)

            if w > 0:
                kv_component = sim / w
            
            unmatched_fmi = {k: fmi[k] for k in unmatched_i}
            unmatched_fmj = {k: fmj[k] for k in unmatched_j}

            mwi = extract_features_model_words(unmatched_fmi)
            mwj = extract_features_model_words(unmatched_fmj)

            hsm_component = compute_hsm_similarity(mwi, mwj)
            tmwm_component = compute_tmwm_similarity(title_i, title_j, alpha, beta, delta = 0.5)

            if tmwm_component == -1:
                theta_1 = m / min(len(fmi), len(fmj))
                theta_2 = 1 - theta_1
                msm_similarity = theta_1*kv_component + theta_2*hsm_component
            else:
                theta_1 = (1-mu) * (m / min(len(fmi), len(fmj)))
                theta_2 = 1 - mu - theta_1
                msm_similarity = theta_1*kv_component + theta_2*hsm_component + mu*tmwm_component
            
            dissimilarity_matrix[i, j] = 1 - msm_similarity
            dissimilarity_matrix[j, i] = 1 - msm_similarity

    print("    -> Clustering start...")

    # Single-linkage clustering
    clusters = [{i} for i in range(n)] # Start with each product alone
    threshold = epsilon

    while True:
        best_pair = None
        best_dist = np.inf

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

    
 