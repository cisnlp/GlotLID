#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from huggingface_hub import snapshot_download

folder = snapshot_download(
    "cis-lmu/glotlid-corpus", 
    repo_type="dataset",
    local_dir="./glotlid-corpus/",
    allow_patterns="v3.1/hbs_Latn/*",
    token="hf_X"
)


# In[ ]:


import glob
import re
import math
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd

# -----------------------
# Config
# -----------------------
paths = {
    "srp": "/nfs/datx/amir/bcms_clean/glotlid-corpus/v3.1/srp_Latn/*",
    "bos": "/nfs/datx/amir/bcms_clean/glotlid-corpus/v3.1/hbs_Latn/*+bos.txt",
    "hrv": "/nfs/datx/amir/bcms_clean/glotlid-corpus/v3.1/hbs_Latn/*+hrv.txt",
}

TOP_N = {
    "srp": 5000,
    "bos": 5000,
    "hrv": 5000,
}

MIN_TOTAL_COUNT = 10   # remove extreme noise
ALPHA = 1.0            # smoothing

# -----------------------
# Helpers
# -----------------------
def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t.isalpha()]

# -----------------------
# Step 1: Count
# -----------------------
print("🚀 Counting...")

lang_counts = {lang: Counter() for lang in paths}
lang_totals = {lang: 0 for lang in paths}
total_counts = Counter()

for lang, pattern in paths.items():
    files = glob.glob(pattern)
    print(f"{lang}: {len(files)} files")

    for file in tqdm(files, desc=f"Processing {lang}"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = tokenize(line)
                if not tokens:
                    continue

                lang_counts[lang].update(tokens)
                total_counts.update(tokens)
                lang_totals[lang] += len(tokens)

# -----------------------
# Debug stats
# -----------------------
print("\n📊 Corpus stats:")
for lang in paths:
    print(f"{lang}: {lang_totals[lang]:,} tokens")

# -----------------------
# Step 2: Compute log-odds
# -----------------------
print("\n🚀 Computing log-odds scores...")

vocab = set(total_counts.keys())

results = {lang: [] for lang in paths}

for word in tqdm(vocab):

    total = total_counts[word]
    if total < MIN_TOTAL_COUNT:
        continue

    for lang in paths:
        count_lang = lang_counts[lang][word]
        total_lang = lang_totals[lang]

        # other languages combined
        count_other = total - count_lang
        total_other = sum(lang_totals.values()) - total_lang

        # add smoothing
        p_lang = (count_lang + ALPHA) / (total_lang + ALPHA * len(vocab))
        p_other = (count_other + ALPHA) / (total_other + ALPHA * len(vocab))

        # log-odds
        score = math.log(p_lang / p_other)

        results[lang].append(
            (word, score, count_lang, total)
        )

# -----------------------
# Step 3: Save top N
# -----------------------
print("\n💾 Saving results...")

for lang in paths:
    data = results[lang]

    df = pd.DataFrame(data, columns=[
        "word",
        "log_odds_score",
        f"{lang}_count",
        "total_count"
    ])

    df = df.sort_values(by="log_odds_score", ascending=False)

    # take top N
    top_n = TOP_N[lang]
    df = df.head(top_n)

    output_file = f"{lang}_top_words.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ {lang}: saved top {len(df)} words")

print("\n🎉 Done!")


# In[ ]:


import glob
import os
import re
import pandas as pd
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
base_paths = {
    "srp": "/nfs/datx/amir/bcms_clean/glotlid-corpus/v3.1/srp_Latn/*",
    "bos": "/nfs/datx/amir/bcms_clean/glotlid-corpus/v3.1/hbs_Latn/*+bos.txt",
    "hrv": "/nfs/datx/amir/bcms_clean/glotlid-corpus/v3.1/hbs_Latn/*+hrv.txt",
}

output_dir = "./filtered_output2"
os.makedirs(output_dir, exist_ok=True)

# -----------------------
# Load vocabularies WITH WEIGHTS
# -----------------------
def load_vocab(lang):
    df = pd.read_csv(f"{lang}_top_words.csv")
    
    # word -> score
    return dict(zip(df["word"], df["log_odds_score"]))

print("📂 Loading vocabularies...")
vocab = {
    "srp": load_vocab("srp"),
    "bos": load_vocab("bos"),
    "hrv": load_vocab("hrv"),
}

# -----------------------
# Tokenizer (LOWERCASE FIX)
# -----------------------
def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t.isalpha()]

# -----------------------
# Weighted classification
# -----------------------
def classify_tokens(tokens):
    scores = {"srp": 0.0, "bos": 0.0, "hrv": 0.0}

    for t in tokens:
        for lang in vocab:
            if t in vocab[lang]:
                scores[lang] += vocab[lang][t]

    total_score = sum(abs(v) for v in scores.values())

    if total_score == 0:
        return {"srp": 0, "bos": 0, "hrv": 0}

    # normalize
    return {
        lang: scores[lang] / total_score
        for lang in scores
    }

# -----------------------
# Better threshold logic
# -----------------------
def is_clean(file_type, ratios):
    best_lang = max(ratios, key=ratios.get)
    best_score = ratios[best_lang]

    # strong dominance
    if best_lang != file_type:
        return False

    if best_score < 0.6:   # dominant enough
        return False

    return True

# -----------------------
# Process files
# -----------------------
print("🚀 Processing files...")

for lang, pattern in base_paths.items():
    files = glob.glob(pattern)

    lang_out_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_out_dir, exist_ok=True)

    for file in tqdm(files, desc=f"Filtering {lang}"):
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]

            clean_lines = []
            mix_lines = []

            for line in lines:
                tokens = tokenize(line)
                ratios = classify_tokens(tokens)

                if is_clean(lang, ratios):
                    clean_lines.append(line)
                else:
                    mix_lines.append(line)

            base_name = os.path.basename(file)
            base_name = base_name.replace('.txt', '')
            
            clean_path = os.path.join(
                lang_out_dir, base_name + '.txt'
            )
            mix_path = os.path.join(
                lang_out_dir, base_name.replace('_Latn_', '_Latn_mix') + '.txt'
            )

            with open(clean_path, "w", encoding="utf-8") as f:
                f.write("\n".join(clean_lines))

            with open(mix_path, "w", encoding="utf-8") as f:
                f.write("\n".join(mix_lines))

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

print("✅ Done!")

