import re

def clean_text(text):

    # Text to str (so non-string inputs don’t error), strip() removes any leading and trailing whitespace, lowercase the string
    text = str(text).lower().strip()

    # ---------- Normalize inch units ----------

    # Replace number + double quote or right double quote with " number inch "
    text = re.sub(r'(\d+)\s*["\u201D]', r'\1inch ', text)                # 29"  → 29inch

    # Replace number + "in", "inch", "inches" (case-insensitive) with " number inch "
    text = re.sub(r'(\d+)\s*(?:inches|inch|in)\b', r'\1inch ', text)    # 29 inches → 29inch

    # ---------- Normalize hertz units ----------

    # Replace number + "hz" (case-insensitive) with " number hz "
    text = re.sub(r'(\d+(?:\.\d+)?)\s*hz\b', r'\1hz ', text)            # 60 hz → 60hz

    # Replace number + "hertz" (case-insensitive) with " number hz "
    text = re.sub(r'(\d+(?:\.\d+)?)\s*hertz\b', r'\1hz ', text)         # 60 hertz → 60hz

    # ---------- Remove spaces / punctuation before units ----------

    text = re.sub(r'[\s\W]+(inch|hz)\b', r'\1 ', text)                  # "-hz" → "hz"

    # ---------- Collapse multiple spaces ----------

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_text_improved(text):

    # Text to str (so non-string inputs don’t error), strip() removes any leading and trailing whitespace, lowercase the string
    text = str(text).lower().strip()

    # -----------------------------------
    # ---------- Standarization ----------
    # -----------------------------------

    # ---------- Standardize inch units ----------

    # Replace number + double quote or right double quote with " number inch "
    text = re.sub(r'(\d+)\s*["\u201D]', r'\1inch ', text)                # 29"  → 29inch

    # Replace number + "in", "inch", "inches" (case-insensitive) with " number inch "
    text = re.sub(r'(\d+)\s*(?:inches|inch|in)\b', r'\1inch ', text)    # 29 inches → 29inch

    # ---------- Standardize hertz units ----------

    # Replace number + "hz" (case-insensitive) with " number hz "
    text = re.sub(r'(\d+(?:\.\d+)?)\s*hz\b', r'\1hz ', text)            # 60 hz → 60hz

    # Replace number + "hertz" (case-insensitive) with " number hz "
    text = re.sub(r'(\d+(?:\.\d+)?)\s*hertz\b', r'\1hz ', text)         # 60 hertz → 60hz

    # --------------------------------------
    # ---------- General cleaning ----------
    # --------------------------------------

    # ---------- Remove non-alphanumeric tokens globally ----------
    
    # Remove ALL non-alphanumerics globally instead of only non-alphanumerics before the unit

    # Step 1: Remove '-' and '/' entirely
    text = re.sub(r'[-/]', '', text)

    # Step 2: Replace all other non-alphanumeric characters with a space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # ---------- Collapse multiple whitespaces ----------

    text = re.sub(r'\s+', ' ', text).strip()

    # Removes all whitespace characters at the start and end
    return text.strip()

def clean_data(data, def_data_cleaner):

    # ---- Clean titles ----

    data['title'] = [def_data_cleaner(t) for t in data['title']]

    # ---- Clean key-value pairs ----

    cleaned_feature_maps = []

    for features in data['featuresMap']:

        cleaned = {k: def_data_cleaner(v) for k, v in features.items()}

        cleaned_feature_maps.append(cleaned)

    data['featuresMap'] = cleaned_feature_maps

    return data