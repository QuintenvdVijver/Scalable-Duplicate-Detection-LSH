import re
import numpy as np
from collections import Counter

def extract_title_model_words(title):

    title_regex = re.compile(r'[A-Za-z0-9]*((?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[A-Za-z0-9]*')

    MW_title = {m.group(0).lower() for m in title_regex.finditer(title) if m.group(0).strip()}

    return MW_title

def extract_features_model_words(featuresMap):

    key_value_regex = re.compile(r'(?:\d+(?:\.\d+)?[a-zA-Z]+$|\d+(?:\.\d+)?$)') 

    MW_features = set()
    if isinstance(featuresMap, dict):
        for value in featuresMap.values():
            string = str(value)

            for token in key_value_regex.findall(string):

                # Check if a the beginning of a string has a numeric pattern
                m = re.match(r'^(\d+(?:\.\d+)?)', token)

                # If yes, returns the numeric substring captured by the regex
                if m:
                    MW_features.add(m.group(1))

    return MW_features

def extract_model_words(data):
  
    MW_title = set()
    MW_features = set() 

    # Iterate over all products
    for row in data.itertuples(index=False):

        # ---------- Extract from title ----------

        title = str(row.title)
        title_matches = extract_title_model_words(title)
        MW_title |= title_matches

        # ---------- Extract from feature values ----------

        featuresMap = row.featuresMap or {}
        feature_matches = extract_features_model_words(featuresMap)
        MW_features |= feature_matches

    return MW_title, MW_features

def build_binary_vectors(data):

    # 1. Build global vocabulary
    MW_title, MW_features = extract_model_words(data)
    MW = sorted (MW_title | MW_features)
    mw_index = {word: idx for idx, word in enumerate(MW)}  # word → position

    print(f"-> title model words found: {len(MW_title)}")
    print(f"-> features words found: {len(MW_features)}")
    print(f"-> total unique model words found: {len(MW)}")

    # 2. Preallocate matrix (num_model_words x num_products)
    binary_vectors = np.zeros((len(MW), len(data)))
    print(f"-> Empty matrix created with dimensions {len(MW)}x{len(data)}")

    # 3. Fill matrix according to Algorithm 1
    for product_index in range(len(data)):

        # Extract title and featuresMap from product
        product= data.iloc[[product_index]]

        # Extract model words for this product

        mw_title, mw_value= extract_model_words(product)

        if product_index == 0:
            print(f"-> Title model words first product: {mw_title}")
            print(f"-> Feature words first product: {mw_value}")

        # If the title or value attribute contains a model words of MW_title
        for w in MW_title:
            if w in mw_title or w in mw_value:
                binary_vectors[mw_index[w], product_index] = 1

        # If the value attribute contains a model words of MW_value
        for w in MW_features:
            if w in mw_value:
                binary_vectors[mw_index[w], product_index] = 1

    return binary_vectors


def extract_relevant_features(data, minimum_frequency):
    feature_freq = Counter()
    for row in data.itertuples(index=False):
        fmap = row.featuresMap or {}
        if isinstance(fmap, dict):
            feature_freq.update(fmap.keys())

    selected_features = {key for key, count in feature_freq.items() if count >= minimum_frequency}

    return selected_features

def extract_model_words_improved(data, selected_features):

    # -------- C. Extract model words --------

    # Unified regex limits the number of model words
    mw_regex = re.compile(r'[A-Za-z0-9]*((?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[A-Za-z0-9]*')
    MW_title = set()
    MW_features = set()

    for row in data.itertuples(index=False):

        # ---- Extract from title ----
        title = str(row.title)
        MW_title.update(
            m.group(0).lower()
            for m in mw_regex.finditer(title)
        )

        # ---- Extract from selected feature values ----
        fmap = row.featuresMap or {}

        for key, value in fmap.items():
            if key not in selected_features:
                continue

            string = str(value)

            MW_features.update(
                m.group(0).lower()
                for m in mw_regex.finditer(string)
            )

    return MW_title, MW_features

def build_binary_vectors_improved(data, relevant_features):

    # 1. Build global vocabulary
    MW_title, MW_features = extract_model_words_improved(data, relevant_features)
    MW = sorted (MW_title | MW_features)
    mw_index = {word: idx for idx, word in enumerate(MW)}  # word → position

    print(f"-> title model words found: {len(MW_title)}")
    print(f"-> features words found: {len(MW_features)}")
    print(f"-> total unique model words found: {len(MW)}")

    # 2. Preallocate matrix (num_model_words x num_products)
    binary_vectors = np.zeros((len(MW), len(data)))
    print(f"-> Empty matrix created with dimensions {len(MW)}x{len(data)}")

    # 3. Fill matrix according to Algorithm 1
    for product_index in range(len(data)):

        # Extract title and featuresMap from product
        product= data.iloc[[product_index]]

        mw_title, mw_value= extract_model_words_improved(product, relevant_features)

        # Print example
        if product_index == 0:
            print(f"-> Title model words first product: {mw_title}")
            print(f"-> Feature words first product: {mw_value}")

        # If the title or value attribute contains a model words of MW_title
        for w in MW_title:
            if w in mw_title or w in mw_value:
                binary_vectors[mw_index[w], product_index] = 1

        # If the value attribute contains a model words of MW_value
        for w in MW_features:
            if w in mw_value:
                binary_vectors[mw_index[w], product_index] = 1

    return binary_vectors