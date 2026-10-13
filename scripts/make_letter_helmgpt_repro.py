# -*- coding: utf-8 -*-
"""Generates an English Letter/Correspondence-style manuscript draft reporting the
reproducibility check of the permeability predictor (Fig. 2a/2c, Spearman r=0.82)
in Xu et al. 2024, HELM-GPT (Bioinformatics 40(6):btae364).

Scope: permeability predictor ONLY (not the KRAS Kd predictor, not Track A/B peptide
design work) - per explicit user decision to submit this as a focused, standalone note.

All numbers are loaded from the project's existing reproduction JSON files, not
hardcoded, consistent with the project's evidence-tracking convention.

Run: .venv_peptide/bin/python analysis/make_letter_helmgpt_repro.py
Output: HELM-GPT_reproducibility_letter_v1.docx
"""
import json
import pathlib

from docx import Document
from docx.shared import Pt

ROOT = pathlib.Path(__file__).resolve().parent.parent

V2 = json.loads((ROOT / "dataset_downloads" / "reproduction_results_v2_spearman.json").read_text())
V3 = json.loads((ROOT / "dataset_downloads" / "reproduction_results_v3_descriptors.json").read_text())
V4 = json.loads((ROOT / "dataset_downloads" / "reproduction_results_v4_h1_h2.json").read_text())
V5 = json.loads((ROOT / "dataset_downloads" / "reproduction_results_v5_permimp_multiseed.json").read_text())
FINAL = json.loads((ROOT / "dataset_downloads" / "final_authors_data_repro.json").read_text())
FINAL_EVAL = json.loads((ROOT / "dataset_downloads" / "final_authors_model_eval.json").read_text())
H1S = V4["H1_summary"]
MS_SC = V5["H2_multiseed_ablation"]["scaffold_summary"]
MS_RD = V5["H2_multiseed_ablation"]["random_summary"]
PI_SC = V5["permutation_importance"]["scaffold"]
PI_RD = V5["permutation_importance"]["random"]


def f3(x):
    return f"{x:.3f}"


doc = Document()
doc.styles["Normal"].font.size = Pt(11)
doc.styles["Normal"].font.name = "Times New Roman"


def h(text, level):
    doc.add_heading(text, level=level)


def p(text, italic=False):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.italic = italic


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for c, v in zip(t.rows[0].cells, header):
        c.text = v
        for r in c.paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = v


# ----------------------------------------------------------------------- title
h("A reproducibility note on the random-forest permeability predictor reported in HELM-GPT", 0)
p("[Author Name(s) — to be completed before submission]")
p("[Affiliation — to be completed before submission]")
p("Corresponding author: [to be completed]")
p(" ")

# ----------------------------------------------------------------------- abstract
h("Abstract", 1)
p(
    "Xu et al. (2024, Bioinformatics 40(6):btae364) introduce HELM-GPT, a generative model for "
    "macrocyclic peptide design, and report that a random-forest (RF) model using ECFP4 fingerprints "
    "plus RDKit descriptors predicts cyclic-peptide membrane permeability (CycPeptMPDB, n=7451) with a "
    "Spearman correlation of r=0.82 on a held-out test set (their Fig. 2c), a figure substantially above "
    "the distribution shown for the same RF configuration in their own Fig. 2a (visually estimated mean "
    f"≈{V5.get('paper_reported_spearman', FINAL['paper_fig2c_final_model_reported'])} vs. "
    f"≈{FINAL['paper_fig2a_RF_FPS_DPS_mean_std'][0]} ± {FINAL['paper_fig2a_RF_FPS_DPS_mean_std'][1]}). "
    "Using the authors' own precomputed feature matrices and their released 10-fold scaffold-split indices "
    "(obtained from the public HELM-GPT repository), we trained a fresh RF model under the same features "
    "and splits and evaluated it across all valid folds. The resulting cross-validated mean "
    f"(Spearman={f3(FINAL['cv_test_spearman_mean'])}±{f3(FINAL['cv_test_spearman_std'])}) closely matches "
    "the distribution in the authors' own Fig. 2a, but no individual fold reaches 0.82 "
    f"(max={f3(max(FINAL['cv_test_spearman_per_fold']))}). We report this discrepancy, together with "
    "supporting permutation-importance and multi-seed ablation checks, to help readers calibrate expectations "
    "when reusing this specific predictor, and to invite clarification from the original authors on the exact "
    "evaluation protocol used to obtain the reported 0.82."
)

# ----------------------------------------------------------------------- background
h("Background", 1)
p(
    "HELM-GPT (Xu et al. 2024) uses an RF permeability predictor to guide reinforcement-learning-based "
    "generation of cell-permeable macrocyclic peptides. The paper states: “We subsequently built a random "
    "forest model for permeability prediction. This model yielded a Spearman correlation coefficient of 0.82 "
    "on the test dataset” (Results, Section 2.2), and Fig. 2c's caption reiterates: “The final predictive "
    "model achieved a Spearman correlation of 0.82.” Separately, Fig. 2a compares eight representation/model "
    "combinations (including the RF on fingerprints+descriptors) with error bars, in which the RF bar is "
    "visibly centered well below 0.82. Because predictor quality directly determines which generated peptides "
    "are advanced in the RL loop, the exact reliability of this specific number matters for anyone reusing the "
    "released model or pipeline."
)

# ----------------------------------------------------------------------- methods
h("Methods", 1)
p(
    "We first reproduced the permeability predictor independently, computing ECFP4 (2048-bit) fingerprints "
    f"and the full set of {V3['n_rdkit_descriptors']} RDKit descriptors for all {V3['total_rows']} CycPeptMPDB "
    "entries ourselves, and training RF models under both scaffold- and random-split protocols. We then swept "
    f"{H1S['n_fits']} hyperparameter/seed/split combinations (H1) and a descriptor-count ablation (H2) to test "
    "whether hyperparameter choice or descriptor selection alone could explain the gap to 0.82."
)
p(
    "To remove any ambiguity from our own feature reconstruction, we subsequently obtained the authors' own "
    "precomputed feature matrices (2048-bit fingerprint block; "
    f"{FINAL['X_dps_shape'][1]}-column descriptor block) and their exact 10-fold scaffold-split index file "
    "directly from the public repository (github.com/charlesxu90/helm-gpt), via its git-lfs storage (the plain "
    "raw.githubusercontent.com endpoint serves only git-lfs pointer files, not the underlying arrays, so the "
    "media.githubusercontent.com/media/... endpoint was used instead). The authors' own saved RF model "
    "(regression_rf.pkl) could not be unpickled under a current scikit-learn release "
    f"({FINAL_EVAL['sklearn_version']}): its serialized decision-tree node array lacks the "
    "‘missing_go_to_left’ field introduced in later scikit-learn versions, indicating it was pickled "
    "under an older (circa 0.23.x) release. We therefore trained a fresh RandomForestRegressor "
    "(n_estimators=300, random_state=42) on the authors' exact features and exact split, which isolates the "
    "comparison to RF hyperparameters/random_state alone — everything else (features, split) is now "
    "identical to the authors' own pipeline."
)
p(
    "As a further robustness check, we recomputed descriptor importance with permutation importance "
    "(scoring=Spearman, held-out test set, n_repeats=10) rather than relying solely on the RF's built-in "
    "impurity-based importances, and re-ran the descriptor-count ablation (k=0,10,25,50,100,217) across five "
    "(split, random_state) seed pairs to test whether differences between descriptor-count settings exceed "
    "seed-to-seed noise."
)

# ----------------------------------------------------------------------- results
h("Results", 1)
table(
    ["Condition", "Val Spearman", "Test Spearman", "Val+Test Spearman"],
    [
        [
            "Authors' reported fold (val_split=7)",
            f3(FINAL["authors_exact_split_val_spearman"]),
            f3(FINAL["authors_exact_split_test_spearman"]),
            f3(FINAL["authors_exact_split_val_plus_test_spearman"]),
        ],
    ],
)
p(
    f"Across all {len(FINAL['cv_test_spearman_per_fold'])} valid folds of the authors' own 10-fold scaffold "
    f"split, test-set Spearman correlation was {f3(FINAL['cv_test_spearman_mean'])} ± "
    f"{f3(FINAL['cv_test_spearman_std'])} (range {f3(min(FINAL['cv_test_spearman_per_fold']))}–"
    f"{f3(max(FINAL['cv_test_spearman_per_fold']))}), closely matching the distribution shown in the authors' "
    f"own Fig. 2a (visually estimated {FINAL['paper_fig2a_RF_FPS_DPS_mean_std'][0]} ± "
    f"{FINAL['paper_fig2a_RF_FPS_DPS_mean_std'][1]}). No individual fold reached the 0.82 reported in Fig. 2c "
    "and the main text, using identical features and identical released splits."
)
p(
    f"In our independently reconstructed feature pipeline, a hyperparameter/seed sweep ({H1S['n_fits']} fits "
    f"across 5 RF configurations x 3 seeds x 2 split types) produced Spearman values in the range "
    f"{f3(H1S['spearman_overall_min'])}–{f3(H1S['spearman_overall_max'])}, which also does not include 0.82 "
    f"(best condition: {H1S['best_condition']}). A descriptor-count ablation showed performance saturating "
    "around the top 25 descriptors (by importance), with no further gain — and a slight decline — from "
    "using the full descriptor set, under either importance ranking:"
)
table(
    ["Split", "Mean Spearman by descriptor count k (0, 10, 25, 50, 100, 217)", "Between-k SD", "Within-k seed SD"],
    [
        ["Scaffold", ", ".join(f3(x) for x in MS_SC["k_means"]), f3(MS_SC["between_k_std"]), f3(MS_SC["mean_within_k_seed_std"])],
        ["Random", ", ".join(f3(x) for x in MS_RD["k_means"]), f3(MS_RD["between_k_std"]), f3(MS_RD["mean_within_k_seed_std"])],
    ],
)
p(
    "Under scaffold splitting, the effect of descriptor count on performance exceeded seed-to-seed noise by "
    f"roughly {f3(MS_SC['between_k_std']/MS_SC['mean_within_k_seed_std'])}x (a genuine effect, concentrated in "
    "the jump from fingerprint-only to the top 10–25 descriptors); under random splitting the same ratio was "
    f"only {f3(MS_RD['between_k_std']/MS_RD['mean_within_k_seed_std'])}x, close to noise level, because "
    f"fingerprint-only features already perform well under random splitting ({f3(MS_RD['k_means'][0])}). "
    "Permutation importance agreed with the RF's built-in (impurity-based) importance on the top two "
    f"descriptors in both split types, but the two rankings diverged considerably further down the list "
    f"(rank correlation {f3(PI_SC['full_rank_correlation_gini_vs_perm'])} for scaffold, "
    f"{f3(PI_RD['full_rank_correlation_gini_vs_perm'])} for random), and the fingerprint block contributed only "
    f"{f3(PI_SC['fingerprint_block_perm_importance_sum']*100)}–{f3(PI_RD['fingerprint_block_perm_importance_sum']*100)}% "
    "of total permutation importance in both cases, confirming descriptors as the dominant predictive signal."
)

# ----------------------------------------------------------------------- discussion
h("Discussion", 1)
p(
    "Using the authors' own features and their own released 10-fold split, we are able to closely reproduce "
    "the distribution shown in the paper's own Fig. 2a, but not the single 0.82 figure quoted in the main text "
    "and Fig. 2c. Because Fig. 2a (a multi-method comparison with error bars) and Fig. 2c (a scatter plot for "
    "one final chosen model) appear in the paper as two distinct analyses, we hypothesize that 0.82 corresponds "
    "to a specific evaluation configuration — e.g. a particular train/test partition, a different "
    "split proportion, or additional hyperparameter tuning — that is not fully specified in the main text "
    "or apparent from the released k-fold index file alone. We did not identify this configuration among the "
    "nine valid folds we were able to construct from the released split file. We want to be explicit that this "
    "is an unresolved hypothesis, not a claim that the reported number is incorrect: important methodological "
    "details (e.g. whether the 0.82 test set is itself one of the ten released folds, additional preprocessing, "
    "or the exact RF hyperparameters, which could not be recovered because the saved model pickle is "
    "incompatible with current scikit-learn) remain undisclosed, and we were not able to consult the original "
    "authors prior to this submission."
)
p(
    "We note that this reproducibility gap concerns only the permeability predictor's reported single-split "
    "performance metric; it does not bear on HELM-GPT's generative/RL methodology, which is the paper's "
    "primary contribution and is unaffected by this note. Our purpose in reporting this is to help other "
    "groups building on the released permeability predictor calibrate their expectations — under a 10-fold "
    "scaffold-split evaluation using the authors' own data, the model's generalization performance should be "
    f"expected to be closer to {f3(FINAL['cv_test_spearman_mean'])} (±{f3(FINAL['cv_test_spearman_std'])}) "
    "than to 0.82 — and to invite the original authors to clarify the exact protocol used to obtain the "
    "reported figure."
)

# ----------------------------------------------------------------------- data availability
h("Data and code availability", 1)
p(
    "All reproduction code, intermediate result files (JSON), and the downloaded authors' feature/split "
    "artifacts used in this note are available at https://github.com/realover82/helm-gpt-reproducibility. The original HELM-GPT code "
    "and data are available at https://github.com/charlesxu90/helm-gpt."
)

h("Conflict of interest", 1)
p("None declared.")

h("References", 1)
p(
    "Xu X, Xu C, He W et al. HELM-GPT: de novo macrocyclic peptide design using generative pre-trained "
    "transformer. Bioinformatics 2024;40(6):btae364. https://doi.org/10.1093/bioinformatics/btae364"
)
p(
    "Li J, Yanagisawa K, Sugita M et al. CycPeptMPDB: a comprehensive database of membrane permeability of "
    "cyclic peptides. J Chem Inf Model 2023;63:2240–50."
)

out = ROOT / "HELM-GPT_reproducibility_letter_v1.docx"
doc.save(out)
print(f"saved: {out}")
