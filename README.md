# HELM-GPT permeability predictor: reproducibility check

This repository accompanies a short reproducibility note (`HELM-GPT_reproducibility_letter_v1.docx`)
checking the Spearman r=0.82 figure reported for the random-forest (RF) permeability predictor in:

> Xu X, Xu C, He W et al. HELM-GPT: de novo macrocyclic peptide design using generative pre-trained
> transformer. *Bioinformatics* 2024;40(6):btae364. https://doi.org/10.1093/bioinformatics/btae364

Original code/data: https://github.com/charlesxu90/helm-gpt

**Scope**: this repo only covers the permeability predictor (their Fig. 2a/2c). It does not cover
the KRAS Kd predictor (Fig. 3) or any other part of the HELM-GPT paper.

## Summary of the finding

Using the authors' own precomputed feature matrices and their released 10-fold scaffold-split
indices, a freshly trained RF (same features, same split, `n_estimators=300, random_state=42`)
gives a cross-validated mean Spearman of **0.574 ± 0.102** across 9 valid folds — closely matching
the distribution shown in the paper's own Fig. 2a (visually estimated 0.547 ± 0.142) — but **no
individual fold reaches the 0.82** reported in Fig. 2c / the main text. See the letter draft for
full discussion, caveats, and the hypothesis we propose (not a claim that the paper is wrong).

## Repository layout

```
scripts/
  reproduce_baseline.py           - initial baseline reproduction attempt
  reproduce_v2_spearman.py        - fingerprint-only RF reproduction (scaffold + random split)
  reproduce_v3_descriptors.py     - fingerprint + self-computed RDKit descriptors (217) reproduction
  reproduce_v4_h1_h2.py           - H1 (hyperparameter/seed sweep) and H2 (descriptor-count ablation)
  reproduce_v5_permimp_multiseed.py - permutation importance + multi-seed descriptor ablation
  final_repro2.py                 - reproduction using the AUTHORS' OWN feature matrices + exact
                                     10-fold scaffold split (the key result in the letter)
  make_letter_helmgpt_repro.py    - generates the letter docx from the result JSONs below

results/
  reproduction_results*.json          - outputs of the scripts above
  final_authors_data_repro.json       - key result: CV on authors' exact data/split
  final_authors_model_eval.json       - documents that the authors' saved model pickle
                                         (regression_rf.pkl) could not be unpickled under current
                                         scikit-learn (built under ~0.23.x; incompatible tree node
                                         dtype) - this is a logged failure, not silently skipped

HELM-GPT_reproducibility_letter_v1.docx - the manuscript draft (Letter/Correspondence style)
```

## Reproducing from scratch

1. Clone the original HELM-GPT repo and locate `data/cpp/`:
   - `X_fps.npy`, `X_dps.npy` (git-lfs tracked — `raw.githubusercontent.com` only serves the LFS
     pointer; fetch the real arrays via
     `https://media.githubusercontent.com/media/charlesxu90/helm-gpt/<ref>/data/cpp/X_fps.npy` etc.)
   - `scaffold_k_fold_idxes.pkl` (10-fold scaffold split indices)
   - `regression_rf.pkl` (their saved model — see `final_authors_model_eval.json` for why it could
     not be loaded under current scikit-learn)
2. Download CycPeptMPDB (7,451 cyclic peptides; Li et al. 2023, *J Chem Inf Model* 63:2240-50) for
   `scripts/reproduce_v2_spearman.py` / `reproduce_v3_descriptors.py` / `reproduce_v4_h1_h2.py`,
   which recompute fingerprints/descriptors independently (used for the H1/H2 ablations and the
   permutation-importance check, which need named RDKit descriptors).
3. Run scripts in the order listed above; each writes its own `reproduction_results_*.json` into
   `results/`, matching the files already included here.
4. `make_letter_helmgpt_repro.py` reads all `results/*.json` and regenerates the letter docx —
   no numbers in the letter are hand-typed.

Large raw data files (CycPeptMPDB CSV, ChEMBL subset, the authors' `X_fps.npy`/`X_dps.npy` arrays,
~130MB total) are intentionally **not** included in this repo to keep it lightweight; they can be
regenerated/re-downloaded per the instructions above.

## Status

Draft reproducibility note, not yet submitted to a journal. Author name/affiliation in the letter
are placeholders pending completion. The original HELM-GPT authors have not yet been contacted
about this finding.
