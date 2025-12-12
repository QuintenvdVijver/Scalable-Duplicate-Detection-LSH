import json
import pandas as pd
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
import os

from data_cleaning import *
from binary_vectors import *
from minhashing import *
from LSH import *
from MSM import *
from evaluation import *


# ------------------------------
# Load data 
# ------------------------------


path_data = "C:/Users/quint/Documents/Master 2025-2026/Block 2/Computer Science for Econometrics/Individual assignment/Assignment code/Data/TVs-all-merged.json"
path_result_train = "C:/Users/quint/Documents/Master 2025-2026/Block 2/Computer Science for Econometrics/Individual assignment/Assignment code/MSMP+ results/train_results.json"
path_results_test = "C:/Users/quint/Documents/Master 2025-2026/Block 2/Computer Science for Econometrics/Individual assignment/Assignment code/MSMP+ results/test_results.json"

# Read json file
with open(path_data, "r", encoding="utf-8") as f: 
    data = json.load(f)

# Flatten lists and convert to dataframe
data = [item for sublist in data.values() for item in sublist]
data = pd.DataFrame(data)

# ------------------------------
# Experiment prep
# ------------------------------


# Set seed
seed = 559692

# Number of bootstraps
n_bootstraps=5

# Data cleaning
print("Cleaning data...")
data = clean_data(data, clean_text)
print(f"-> Loaded and cleaned {len(data)} product entries.")

# Binary representation for LSH
print("Creating the binary vectors...")
binary_vectors = build_binary_vectors(data)
print("-> Binary vectors succesfully created")

# Minhash signatures
print("Creating the minhash signatures...")
n_permutations = len(binary_vectors) // 2
minhash_signatures = build_minhash_signatures(binary_vectors, n_permutations, seed)
print(f"-> Minhash signatures created succesfully with dimensions {len(minhash_signatures)}x{len(minhash_signatures[0])}")

# Bootstrap samples for evaluation
print("Generating bootstraps...")
bootstrap_samples, bootstrap_test_samples = build_bootstrap_samples(data, number_samples = n_bootstraps, seed = seed)
print(f"-> Succesfully generated {len(bootstrap_samples)} bootstrap samples with {len(bootstrap_samples[0])} products each.")

# True pairs for evaluation
print("Finding the true pairs...")
true_pairs = get_true_pairs(data)
print(f"-> {len(true_pairs)} true pairs found.")


# # ------------------------------
# # Main experiment
# # ------------------------------


# # -------------- Hyperparameter setup ---------------


# Hyperparameter setup
print("Setting up hyperparameter optimization...")

band_configs = []
    # allowed to deviate from n = b*r slightly, it only make the S-curve slightly smoother at the threshold, adds a softer tail,
    #   the contribution of the last band to collission probability is slightly different but it cannot invalidate the LSH process
for r in range(1,n_permutations+1): # Number of rows between 1 and n_permutations
    b = (n_permutations + r - 1) // r # Ceiling division: ensures we have enough bands for all rows
    max_deficit = 3

    used_rows = r * b
    deficit = used_rows - n_permutations

    if 0 <= deficit <= max_deficit:
            band_configs.append((r, b))
print(f"- {len(band_configs)} band configurations: {band_configs}")


# # -------------- Experiment ---------------


# Storage for evaluation (either append to existing results or create empty list)
if os.path.exists(path_result_train):
    with open(path_result_train, "r", encoding="utf-8") as f: 
        train_results = json.load(f)
        print("TRAIN_RESULTS JSON LOADED")
else:
    train_results = []
    print("TRAIN_RESULTS INITIALIZED AS EMPTY")

if os.path.exists(path_results_test):
    with open(path_results_test, "r", encoding="utf-8") as f: 
        test_results = json.load(f)
        print("TEST_RESULTS JSON LOADED")
else:
    test_results = []
    print("TEST_RESULTS INITIALIZED AS EMPTY")


# Run experiment
for b in range(n_bootstraps):

    print('============================================')
    print(f'=============== Bootstrap {b} ================')
    print('============================================')

    train_indices = list(bootstrap_samples[b].index)
    test_indices = list(bootstrap_test_samples[b].index)

    train_data = bootstrap_samples[b]
    test_data = bootstrap_test_samples[b]

    # True pairs for evaluation (pair if both entries are in corresponding indices set)
    true_pairs_train = [
        pair for pair in true_pairs
        if pair[0] in train_indices and pair[1] in train_indices
    ]
    # print(f"-> True pair count: {len(true_pairs_train)}")
    true_pairs_test = [
        pair for pair in true_pairs
        if pair[0] in test_indices and pair[1] in test_indices
    ]
    print(f"-> True pair count: {len(true_pairs_test)}")

    # Binary representation for LSH
    binary_vectors_train = binary_vectors[:, train_indices]
    binary_vectors_test  = binary_vectors[:, test_indices]

    # Minhash signatures
    minhash_signatures_train = minhash_signatures[:, train_indices]
    minhash_signatures_test = minhash_signatures[:, test_indices]


    # -------------- Optimization ---------------


    for (rows, bands) in band_configs:

        # Skip iteration if already done
        if any(results["bootstrap"] == b and results["(r,b)"] == [rows,bands] for results in test_results):
            print(f"Skipping bootstrap {b}, (r,b)=({rows},{bands}) — already done")
            continue

        print('========================================')
        print(f"(r,b) = ({rows},{bands})")
        print('========================================')


        # -------------- Run optimization ---------------


        # Create objective function
        optuna_objective = make_objective(train_indices, train_data, minhash_signatures_train, true_pairs_train, rows, bands)

        # Set sampler and pruner
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

        # Execute Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
            )
        study.optimize(optuna_objective, n_trials=10)


        # -------------- Store optimal configuration for each [bootstrap, (r,b)] pair ---------------


        best = study.best_trial

        train_results.append({
            "bootstrap": b,
            "(r,b)": [rows, bands],
            "best_f1": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials)
        })


        # -------------- Run optimal configurations on test set ---------------


        # Perform LSH
        print('=========================')
        print("Running best configuration on test set...")
        print(f"-> performing LSH... [{datetime.now().strftime('%H:%M:%S')}]")
        candidate_pairs = perform_LSH(minhash_signatures_test, rows, bands)
        print(f"    -> Succesfully found {len(candidate_pairs)} candidate pairs [{datetime.now().strftime('%H:%M:%S')}]")

        # Perform MSM
        print(f"-> performing MSM... [{datetime.now().strftime('%H:%M:%S')}]")
        clusters = perform_MSM(
            test_data, 
            candidate_pairs, 
            alpha=best.params["alpha"],
            beta=best.params["beta"],
            gamma=best.params["gamma"],
            mu=best.params["mu"],
            epsilon=best.params["epsilon"]
            )

        clusters = perform_MSM(
            test_data, 
            candidate_pairs, 
            alpha=0.602, 
            beta=0.000, 
            gamma=0.756, 
            mu=0.650,
            epsilon=best.params["epsilon"]
            )


        # -------------- Store test results ---------------


        print(f"-> Appending results... [{datetime.now().strftime('%H:%M:%S')}]")
        print('=========================')

        # Convert indices back to original indices
        candidate_pairs_global = {
            tuple(sorted((test_indices[i], test_indices[j])))
            for (i, j) in candidate_pairs
        }

        cluster_pairs = []
        for cluster in clusters:
            items = list(cluster)
            if len(items) > 1:
                # all combinations of 2 within each cluster
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair_test_indices = tuple(
                            sorted((test_indices[items[i]], test_indices[items[j]]))
                        )
                        cluster_pairs.append(pair_test_indices)

        total_duplicates = len(true_pairs_test)
        print(f"total duplicates: {total_duplicates}")
        n_test_unique = len(set(test_indices))
        duplicates_found_lsh = len(candidate_pairs_global & set(true_pairs_test))
        print(f"duplicates_found_lsh: {duplicates_found_lsh}")
        num_comparisons_lsh = len(candidate_pairs_global)

        pq_lsh = compute_pair_quality(duplicates_found_lsh, num_comparisons_lsh)
        pc_lsh = compute_pair_completeness(duplicates_found_lsh, total_duplicates)
        print(f"pc_lsh: {pc_lsh}")
        f1_star = compute_F1(pq_lsh, pc_lsh)
        foc_lsh = compute_fraction_of_comparisons(num_comparisons_lsh, n_test_unique)

        duplicates_found_msm = len(set(true_pairs_test) & set(cluster_pairs))
        num_comparisons_msm = len(candidate_pairs)
        pq_msm = compute_pair_quality(duplicates_found_msm, num_comparisons_msm)
        pc_msm = compute_pair_completeness(duplicates_found_msm, total_duplicates)
        f1 = compute_F1(pq_msm, pc_msm)
        foc_msm = compute_fraction_of_comparisons(num_comparisons_msm, n_test_unique)

        test_results.append({
            "bootstrap": b,
            "(r,b)": [rows,bands],
            "t": (1/bands)**(1/rows),
            "candidate_pairs": len(candidate_pairs),
            "duplicates_found_lsh": duplicates_found_lsh,
            "duplicates_found_msm": duplicates_found_msm,
            "foc_lsh": foc_lsh,
            "pq_lsh": pq_lsh,
            "pc_lsh": pc_lsh,
            "f1_star": f1_star,
            "foc_msm": foc_msm,
            "pq_msm": pq_msm,
            "pc_msm": pc_msm,
            "f1": f1
        })

        # Update json files after full iteration
        with open(os.path.join("MSMP+ results", "train_results.json"), "w") as f:
            json.dump(train_results, f, indent=4)

        with open(os.path.join("MSMP+ results", "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)


# --------------- Average the evaluation metrics of the matched results ---------------


# Load results json
with open(path_results_test, "r", encoding="utf-8") as f: 
    results_json = json.load(f)


# Convert results_json to pd dataframe
test_results_df = pd.DataFrame(results_json)
test_results_df["(r,b)"] = test_results_df["(r,b)"].apply(tuple) # Convert list to tuple for .agg

summary_df = (
    test_results_df
    .groupby(["(r,b)", "t"], as_index=False)
    .agg({
        "candidate_pairs": "mean",
        "duplicates_found_lsh": "mean",
        "duplicates_found_msm": "mean",
        "foc_lsh": "mean",
        "pq_lsh": "mean",
        "pc_lsh": "mean",
        "f1_star": "mean",
        "foc_msm": "mean",
        "pq_msm": "mean",
        "pc_msm": "mean",
        "f1": "mean"
    }).sort_values("foc_lsh", ascending=True)
)

print('============================================')
print('================= Results ==================')
print('============================================')

print(summary_df)


# --------------- Construct the relevant graphs (cluster results based on FOC) ---------------


# Pair Completeness LSH
plt.figure(figsize=(8,5))
plt.plot(summary_df["foc_lsh"], summary_df["pc_lsh"], label='MSMP+', color='grey')
plt.xlabel("Fraction of comparisons", fontsize=12)
plt.ylabel("Pair completeness", fontsize=12)
plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.ylim(-0.01, 1)
plt.xlim(-0.01, 1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join("MSMP+ results", "pair_completeness_lsh.png"), dpi=300)
plt.close()

# Pair Quality LSH
plt.figure(figsize=(8,5))
plt.plot(summary_df["foc_lsh"], summary_df["pq_lsh"], label='MSMP+', color='grey')
plt.xlabel("Fraction of comparisons", fontsize=12)
plt.ylabel("Pair quality", fontsize=12)
plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.xlim(-0.01, 0.2)
plt.ylim(-0.01, 0.2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join("MSMP+ results", "pair_quality_lsh.png"), dpi=300)
plt.close()

# F1 star
plt.figure(figsize=(8,5))
plt.plot(summary_df["foc_lsh"], summary_df["f1_star"], label='MSMP+', color='grey')
plt.xlabel("Fraction of comparisons", fontsize=12)
plt.ylabel("F1*-measure", fontsize=12)
plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.ylim(bottom=0)
plt.xlim(-0.01, 1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join("MSMP+ results", "f1_star_lsh.png"), dpi=300)
plt.close()

# F1 MSM
plt.figure(figsize=(8,5))
plt.plot(summary_df["foc_lsh"], summary_df["f1"], label='MSMP+', color='grey')
plt.xlabel("Fraction of comparisons", fontsize=12)
plt.ylabel("F1-measure", fontsize=12)
plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.ylim(bottom=0)
plt.xlim(-0.01, 1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join("MSMP+ results", "f1.png"), dpi=300)
plt.close()