from collections import Counter
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import MODELS

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
    vocab.difference_update({"man", "male", "husband", "woman", "female", "wife", "higher-pitched", "lower-pitched", "higher-pitch", "lower-pitch", "high-pitched", "low-pitched", "high-pitch", "low-pitch", "higher-pitched-voice", "lower-pitched-voice", "higher-pitch-voice", "lower-pitch-voice", "high-pitched-voice", "low-pitched-voice", "high-pitch-voice", "low-pitch-voice"})

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

def collect_all_adjectives(df):
    adjectives_fem = []
    adjectives_masc = []
    for _, row in df.iterrows():
        adjs_a = row["adjectives_a"]
        adjs_b = row["adjectives_b"]

        try:

            adjs_a = json.loads(adjs_a.replace("'", '"'))
            assert isinstance(adjs_a, list)
        except:
            adjs_a = []
        try:
            adjs_b = json.loads(adjs_b.replace("'", '"'))
            assert isinstance(adjs_b, list)
        except:
            adjs_b = []
        
        if row["gender_1"] == "female":
            adjectives_fem.extend(adjs_a)
            adjectives_masc.extend(adjs_b)
        else:
            adjectives_fem.extend(adjs_b)
            adjectives_masc.extend(adjs_a)

    return adjectives_fem, adjectives_masc


def main(model_name, only_area=None):
    model_short = model_name.split("/")[-1]
    df = pd.read_csv(f"evaluation/gender_bias/outputs/{model_short}_with_adjectives.csv")

    if only_area is not None:
        df = df[df["area"] == only_area]

    num_areas = len(df["area"].unique())
    plot_data = []
    max_words = 0
    for area in df["area"].unique():
        df_area = df[df["area"] == area]
        adjectives_fem, adjectives_masc = collect_all_adjectives(df_area)

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
        final_df = final_df[abs(final_df['odds_ratio']) > 1.5]
        final_df = final_df[abs(final_df['c_a']) > 10]
        final_df = final_df[abs(final_df['c_b']) > 10]
        final_df = final_df.sort_values(by="delta", ascending=True)

        max_words = max(max_words, len(final_df))
        plot_data.append((area, final_df))

    fig_height = 2.5
    fig, axes = plt.subplots(1, num_areas, figsize=(3 * num_areas, fig_height))
    for iarea, (area, final_df) in enumerate(plot_data):
        ax = axes[iarea] if num_areas > 1 else axes
        if only_area is None:
            ax.set_title(area, fontsize=10)

        max_abs_delta = final_df["delta"].abs().max()
        if pd.isna(max_abs_delta) or max_abs_delta == 0:
            max_abs_delta = 1.0
        max_abs_delta *= 1.2

        final_df["delta_plot"] = final_df["delta"].clip(-max_abs_delta, max_abs_delta)

        print(f"Area: {area}, num words: {len(final_df)}, max_abs_delta: {max_abs_delta:.2f}")
        row_step = 1/len(final_df)
        y = np.arange(len(final_df)) * row_step
        final_df["y"] = y

        # Simple horizontal bars
        colors = ["#f44cb0"]*(final_df["delta_plot"] <= 0).sum() + ['#3b82f6']*(final_df["delta_plot"] > 0).sum()
        ax.barh(final_df["y"], final_df["delta_plot"], alpha=0.8, height=min(0.85/len(final_df), 0.85/3), color=colors)
        
        # Zero reference line
        ax.axvline(0, linewidth=1, color='black', linestyle='-', alpha=0.5)
        ax.set_xlim(-max_abs_delta, max_abs_delta)
        
        # Labels
        ax.set_yticks(final_df["y"])
        ax.set_yticklabels(final_df["word"], fontsize=7)
        if len(final_df) > 0:
            ax.set_ylim(y[-1] + row_step * 0.6, -row_step * 0.6)
        ax.invert_yaxis()
        
        
        ax.set_xlabel("Association with masculine\nover feminine voices (LOR)", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            

    fig.tight_layout()
    os.makedirs("evaluation/gender_bias/plots", exist_ok=True)
    fig.savefig(f"evaluation/gender_bias/plots/fighting_words_{model_short}{'_' + only_area if only_area else ''}.pdf", bbox_inches="tight", pad_inches=0.01)

if __name__ == "__main__":
    for model_name in MODELS:
        for prefix in ("best-lora_", ""):        
            main(model_name=prefix + model_name) #, only_area="science")