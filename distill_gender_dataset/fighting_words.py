from collections import Counter
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
import os

# from https://github.com/TAI-HAMBURG/SpeechAccess-Bias/blob/main/code/prompts.py
def fighting_words_from_tokens(tokens_a, tokens_b, alpha=1, min_count=1):
    """
    tokens_a / tokens_b: lists of tokens (e.g., ["clean","energy","clean",...])
    alpha: total prior mass for informative Dirichlet prior (>=0, try 0.1--10)
    min_count: drop words whose total count across A+B < min_count
    Returns DataFrame with columns: word, c_a, c_b, p_a, p_b, delta, z
      +z => distinctive for A; -z => for B.
    """
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    Na, Nb = sum(ca.values()), sum(cb.values())
    vocab = set(ca) | set(cb)

    # remove trivial words like man, male...
    # vocab.difference_update({"man", "male", "husband", "woman", "female", "wife", "she", "her", "he", "him", "his"})

    if min_count > 1:
        vocab = {w for w in vocab if ca.get(w,0) + cb.get(w,0) >= min_count}
    if not vocab:
        return pd.DataFrame(columns=["word","c_a","c_b","p_a","p_b","delta","z"])

    pooled_total = sum(ca.get(w,0) + cb.get(w,0) for w in vocab) or 1

    rows = []
    for w in vocab:
        ya, yb = ca.get(w,0), cb.get(w,0)
        a0w = alpha * (ya + yb) / pooled_total  # informative Dirichlet prior per word

        numer_a = ya + a0w
        denom_a = (Na + alpha) - numer_a
        numer_b = yb + a0w
        denom_b = (Nb + alpha) - numer_b

        if min(numer_a, denom_a, numer_b, denom_b) <= 0:
            continue

        logit_a = np.log(numer_a) - np.log(denom_a)
        logit_b = np.log(numer_b) - np.log(denom_b)
        delta = logit_a - logit_b
        var = 1.0/numer_a + 1.0/denom_a + 1.0/numer_b + 1.0/denom_b
        z = delta / np.sqrt(var)

        rows.append((w, ya, yb, numer_a/(Na+alpha), numer_b/(Nb+alpha), delta, z))

    return (pd.DataFrame(rows, columns=["word","c_a","c_b","p_a","p_b","delta","z"])
              .sort_values("z", ascending=False)
              .reset_index(drop=True))

def collect_all_adjectives(csv_path):
    adjectives_fem = []
    adjectives_masc = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            gender = row[2]
            text = row[1]
            tokens = re.findall(r'\b\w+\b', text.lower())
            match gender:
                case "F":
                    adjectives_fem.extend(tokens)
                case "M":
                    adjectives_masc.extend(tokens)

    return adjectives_fem, adjectives_masc

def main(model_name = "Qwen3-Omni-30B-A3B-Instruct"):
    model_short = model_name.split("/")[-1]
    csv_dir = "distill_gender_dataset/datasets/"
    for file in os.listdir(csv_dir):
        if not file.startswith(model_short) or not file.endswith(".csv"):
            continue
        
        prompt_key = file[len(model_short)+1:-4]  # remove model_short_ and .csv
        csv_path = os.path.join(csv_dir, file)

        ax = plt.gca()
        ax.set_title("Fighting Words", fontsize=10)

        adjectives_fem, adjectives_masc = collect_all_adjectives(csv_path)

        determining_words_female = fighting_words_from_tokens(adjectives_fem, adjectives_masc, alpha=1, min_count=5)
        determining_words_female['gender'] = 'female'
        determining_words_female['odds_ratio'] = np.exp(determining_words_female['delta'])
        determining_words_female['delta'] = - determining_words_female['delta']
        determining_words_male = fighting_words_from_tokens(adjectives_masc, adjectives_fem, alpha=1, min_count=5)
        determining_words_male['odds_ratio'] = np.exp(determining_words_male['delta'])
        determining_words_male['gender'] = 'male'

        n = 5
        determining_words_male = determining_words_male.nlargest(n, 'z')
        determining_words_female = determining_words_female.nlargest(n, 'z')


        final_df = pd.concat([determining_words_female, determining_words_male])

        final_df = final_df[final_df['z'] > 1.96]
        # final_df = final_df[abs(final_df['odds_ratio']) > 1.5]
        # final_df = final_df[abs(final_df['c_a']) > 20]
        # final_df = final_df[abs(final_df['c_b']) > 20]


        final_df = final_df.sort_values(by="delta", ascending=False)
        print(final_df)

        y = np.arange(len(final_df))
        final_df["y"] = y

        # Simple horizontal bars
        ax.barh(final_df["y"], final_df["delta"], alpha=0.8, height=0.7)
        
        # Zero reference line
        ax.axvline(0, linewidth=1, color='black', linestyle='-', alpha=0.5)
        
        # Labels
        ax.set_yticks(final_df["y"])
        ax.set_yticklabels(final_df["word"], fontsize=7)
        ax.invert_yaxis()
        
        
        ax.set_xlabel("Association with masculine\nover feminine voices (LOR)", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            

        plt.tight_layout()
        plt.savefig(f"distill_gender_dataset/outputs/fighting_words_{model_short}_{prompt_key}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close()
if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model_name)
    else:
        parser = argparse.ArgumentParser(description="Generate rejection emails based on audio recordings.")
        parser.add_argument("--model_name", type=str, default="Qwen/best-checkpoint_Qwen3-Omni-30B-A3B-Instruct", help="Model name or path to the model checkpoint.") # Qwen3-Omni-30B-A3B-Instruct
        args = parser.parse_args()
        main(args.model_name)