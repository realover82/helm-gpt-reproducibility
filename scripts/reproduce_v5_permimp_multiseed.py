"""v59 Track B 추가 검증: permutation importance로 descriptor 중요도 재확인 +
   descriptor ablation(k)을 다중 시드로 재실행해 k 사이 차이가 시드 잡음과 구분되는지 확인.

   기반: v4(reproduce_v4_h1_h2.py)의 self-computed ECFP4(2048)+RDKit descriptor(217종) 파이프라인과
   동일한 feature/split 구성을 재사용한다. 저자 제공 X_dps.npy.try(211열, cpp_authors/)는 현재 RDKit
   217종과 개수가 달라 열-이름 매핑을 신뢰할 수 없으므로(Rule 6: 불명확한 identifier 임의 매핑 금지),
   descriptor 이름이 확실한 self-computed 파이프라인에서만 permutation importance 이름을 보고한다.

   H2-perm: RF built-in feature_importances_(Gini, train-set 편향) 대신
            sklearn permutation_importance(held-out test set, scoring=Spearman)로 top descriptor 재확인.
   H2-multiseed: permutation importance 기준으로 seed=42 split 1회에서만 top-k descriptor 집합을 확정한 뒤,
            k값별로 (split+RF)seed를 5개(42,7,2024,123,999, H1과 동일 관례)로 반복해
            "k차이" 분산과 "동일 k 내 시드" 분산을 비교한다.
   실행: .venv_peptide/bin/python dataset_downloads/reproduce_v5_permimp_multiseed.py
   출력: reproduction_results_v5_permimp_multiseed.json (+ repro_v5.log)
"""
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

RDLogger.DisableLog("rdApp.*")

T0 = time.time()
report = {"experiment": "v5_permimp_multiseed", "paper_reported_spearman": 0.82}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- data (v4와 동일)
df = pd.read_csv("dataset_downloads/cycpeptmpdb_all.csv")
df = df[df["smi"].apply(lambda s: Chem.MolFromSmiles(s) is not None)].reset_index(drop=True)
report["total_rows"] = len(df)
log(f"rows={len(df)}")


def sm_to_ecfp4(smiles, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


desc_list = Descriptors._descList
desc_names = [n for n, _ in desc_list]
report["n_rdkit_descriptors"] = len(desc_names)


def sm_to_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    vals = []
    for _, func in desc_list:
        try:
            vals.append(func(mol))
        except Exception:
            vals.append(np.nan)
    return vals


X_fp = np.stack([sm_to_ecfp4(s) for s in df["smi"]])
X_desc = np.nan_to_num(np.array([sm_to_desc(s) for s in df["smi"]]), nan=0.0, posinf=0.0, neginf=0.0)
X_all = np.concatenate([X_fp, X_desc], axis=1)
y = df["score"].values.astype(float)
log(f"features built: fp={X_fp.shape} desc={X_desc.shape}")

scaffold_to_idx = defaultdict(list)
for idx, smi in enumerate(df["smi"]):
    scaffold_to_idx[MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), includeChirality=False)].append(idx)
SCAFFOLD_CLUSTERS = list(scaffold_to_idx.values())
N = len(df)
N_TRAIN, N_VAL = int(N * 0.8), int(N * 0.1)


def split_indices(kind, seed):
    """v3/v4와 동일한 분할 로직. seed가 split과 RF random_state 둘 다에 쓰인다(H1과 동일 관례)."""
    rng = np.random.RandomState(seed)
    if kind == "random":
        perm = rng.permutation(N)
        return perm[:N_TRAIN], perm[N_TRAIN + N_VAL:]
    clusters = list(SCAFFOLD_CLUSTERS)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)
    train, val, test = [], [], []
    for c in clusters:
        if len(train) + len(c) <= N_TRAIN:
            train.extend(c)
        elif len(val) + len(c) <= N_VAL:
            val.extend(c)
        else:
            test.extend(c)
    return np.array(train), np.array(test)


def spearman_scorer(estimator, X, y_true):
    pred = estimator.predict(X)
    return spearmanr(y_true, pred).correlation


SEEDS_H1 = [42, 7, 2024]
SEEDS_EXTRA = [123, 999]
SEEDS_ALL = SEEDS_H1 + SEEDS_EXTRA
K_GRID = (0, 10, 25, 50, 100, len(desc_names))

perm_results = {}
h2_multiseed = {}

for split_kind in ("scaffold", "random"):
    # ---- 1) 고정 기준(seed=42)으로 모델 학습 + 두 importance 계산 ----
    tr, te = split_indices(split_kind, 42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_all[tr], y[tr])

    gini_imp_all = model.feature_importances_
    gini_imp = gini_imp_all[X_fp.shape[1]:]
    gini_order = np.argsort(gini_imp)[::-1]

    log(f"{split_kind}: permutation_importance on held-out test (n_repeats=10)...")
    pi = permutation_importance(
        model, X_all[te], y[te],
        scoring=spearman_scorer,
        n_repeats=10, random_state=42, n_jobs=-1,
    )
    desc_pi_mean = pi.importances_mean[X_fp.shape[1]:]
    desc_pi_std = pi.importances_std[X_fp.shape[1]:]
    fp_pi_mean = pi.importances_mean[:X_fp.shape[1]]
    perm_order = np.argsort(desc_pi_mean)[::-1]

    top25_gini_set = set(gini_order[:25].tolist())
    top25_perm_set = set(perm_order[:25].tolist())
    overlap = len(top25_gini_set & top25_perm_set)
    rank_corr = spearmanr(gini_imp, desc_pi_mean).correlation

    perm_results[split_kind] = {
        "fingerprint_block_perm_importance_sum": float(np.sum(fp_pi_mean)),
        "descriptor_block_perm_importance_sum": float(np.sum(desc_pi_mean)),
        "top25_by_permutation_importance": [
            {
                "name": desc_names[i],
                "perm_importance_mean": float(desc_pi_mean[i]),
                "perm_importance_std": float(desc_pi_std[i]),
                "gini_rank": int(np.where(gini_order == i)[0][0]) + 1,
            }
            for i in perm_order[:25]
        ],
        "top25_by_gini_importance_for_comparison": [
            {
                "name": desc_names[i],
                "gini_importance": float(gini_imp[i]),
                "perm_rank": int(np.where(perm_order == i)[0][0]) + 1,
            }
            for i in gini_order[:25]
        ],
        "top25_overlap_gini_vs_perm": overlap,
        "full_rank_correlation_gini_vs_perm": float(rank_corr),
    }
    log(f"{split_kind}: top25 overlap(gini vs perm)={overlap}/25, full rank corr={rank_corr:.3f}")

    # ---- 2) permutation importance 순서를 1회 고정해 k-ablation을 다중 시드로 재실행 ----
    ms = {}
    for k in K_GRID:
        runs = []
        for seed in SEEDS_ALL:
            tr_s, te_s = split_indices(split_kind, seed)
            X_k = X_fp if k == 0 else np.concatenate([X_fp, X_desc[:, perm_order[:k]]], axis=1)
            rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X_k[tr_s], y[tr_s])
            pred = rf.predict(X_k[te_s])
            sp = spearmanr(y[te_s], pred).correlation
            runs.append({"seed": seed, "spearman": float(sp)})
            log(f"H2multiseed {split_kind}/k={k}/seed={seed}: spearman={sp:.4f}")
        sp_vals = [r["spearman"] for r in runs]
        ms[f"k{k}"] = {
            "runs": runs,
            "spearman_mean": float(np.mean(sp_vals)),
            "spearman_std": float(np.std(sp_vals, ddof=1)),
            "spearman_min": float(np.min(sp_vals)),
            "spearman_max": float(np.max(sp_vals)),
        }
    h2_multiseed[split_kind] = ms

    k_means = np.array([ms[f"k{k}"]["spearman_mean"] for k in K_GRID])
    within_k_stds = np.array([ms[f"k{k}"]["spearman_std"] for k in K_GRID])
    between_k_std = float(np.std(k_means, ddof=1))
    mean_within_k_std = float(np.mean(within_k_stds))
    h2_multiseed[f"{split_kind}_summary"] = {
        "k_grid": list(K_GRID),
        "k_means": k_means.tolist(),
        "between_k_std": between_k_std,
        "mean_within_k_seed_std": mean_within_k_std,
        "between_exceeds_within": bool(between_k_std > mean_within_k_std),
    }
    log(f"{split_kind} summary: between_k_std={between_k_std:.4f} vs mean_within_k_seed_std={mean_within_k_std:.4f}")

report["permutation_importance"] = perm_results
report["H2_multiseed_ablation"] = h2_multiseed
report["seeds_used"] = SEEDS_ALL
report["note"] = (
    "permutation_importance는 scoring=Spearman으로 held-out test set에서 n_repeats=10으로 계산했다 "
    "(RF built-in feature_importances_는 Gini/MSE-감소 기반이라 연속형 descriptor를 이진 fingerprint "
    "비트보다 과대평가하는 경향이 있어 대조군으로 함께 보고한다). descriptor ranking은 scaffold/random "
    "split 각각 seed=42 결과로 1회만 고정한 뒤 k-ablation 전체에 재사용했다(같은 k가 seed마다 다른 "
    "descriptor 집합을 쓰면 비교가 깨지므로). H2_multiseed_ablation은 동일 k에서 (split, RF random_state) "
    "시드 쌍을 5개(42,7,2024,123,999 - H1의 3개 시드에 2개 추가)로 바꿔 반복한 결과이며, k 평균들의 "
    "표준편차(between_k_std)와 동일 k 내 시드 표준편차의 평균(mean_within_k_seed_std)을 비교해 k효과가 "
    "시드 잡음보다 큰지(between_exceeds_within) 판단한다. 시드가 split과 RF random_state에 동시에 적용되므로 "
    "이 noise 추정치는 두 요인을 합친 보수적(상한) 추정이다."
)
report["elapsed_sec"] = round(time.time() - T0, 1)

print(json.dumps({k: v for k, v in report.items() if k not in ("permutation_importance", "H2_multiseed_ablation")}, indent=2, ensure_ascii=False))
with open("dataset_downloads/reproduction_results_v5_permimp_multiseed.json", "w") as f:
    json.dump(report, f, indent=2)
log("saved reproduction_results_v5_permimp_multiseed.json")
