"""v55용 재현 실험: v54가 남긴 잔여 격차(최선 0.695 vs 논문 0.82)의 원인 분리.

H1 (하이퍼파라미터/시드): 동일 feature(ECFP4+descriptor)로 하이퍼파라미터 조합 x 다중 시드를
     반복해 Test Spearman의 분포(평균+-표준편차, min/max)를 산출한다.
     질문: 논문 보고값 0.82가 이 분포 안에 들어오는가?

H2 (descriptor ablation): RF feature importance 기준 상위 k개 descriptor만 사용했을 때
     전체 217종 사용 대비 성능이 어떻게 변하는가?
     질문: 전체 descriptor 사용이 오히려 노이즈를 더하는가?

선행 조건은 reproduce_v3_descriptors.py와 동일하게 유지하고, 바꾸는 축만 명시적으로 바꾼다.
출력: reproduction_results_v4_h1_h2.json (+ repro_v4.log)
실행: .venv_peptide/bin/python dataset_downloads/reproduce_v4_h1_h2.py
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
from sklearn.metrics import r2_score, root_mean_squared_error

RDLogger.DisableLog("rdApp.*")

T0 = time.time()
report = {"experiment": "v4_H1_H2", "paper_reported_spearman": 0.82}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- data
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
    """v3 스크립트와 동일한 분할 로직. seed만 외부에서 주입한다."""
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


def fit_eval(X, tr, te, seed, params, return_model=False):
    model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    res = {
        "spearman": float(spearmanr(y[te], pred).correlation),
        "r2": float(r2_score(y[te], pred)),
        "rmse": float(root_mean_squared_error(y[te], pred)),
    }
    return (res, model) if return_model else res


# ---------------------------------------------------------------- H1
CONFIGS = {
    "v3_baseline_n300": {"n_estimators": 300},
    "n600": {"n_estimators": 600},
    "n600_maxfeat0.1": {"n_estimators": 600, "max_features": 0.1},
    "n600_maxfeat0.3": {"n_estimators": 600, "max_features": 0.3},
    "n600_leaf2_maxfeat0.3": {"n_estimators": 600, "max_features": 0.3, "min_samples_leaf": 2},
}
SEEDS = [42, 7, 2024]

h1 = {}
for split_kind in ("scaffold", "random"):
    for cfg_name, params in CONFIGS.items():
        runs = []
        for seed in SEEDS:
            tr, te = split_indices(split_kind, seed)
            r = fit_eval(X_all, tr, te, seed, params)
            r["seed"] = seed
            r["n_test"] = int(len(te))
            runs.append(r)
            log(f"H1 {split_kind}/{cfg_name}/seed{seed}: spearman={r['spearman']:.4f}")
        sp = [r["spearman"] for r in runs]
        h1[f"{split_kind}__{cfg_name}"] = {
            "params": params,
            "runs": runs,
            "spearman_mean": float(np.mean(sp)),
            "spearman_std": float(np.std(sp, ddof=1)),
            "spearman_min": float(np.min(sp)),
            "spearman_max": float(np.max(sp)),
        }
report["H1_hyperparameter_seed_distribution"] = h1

best_overall = max(h1.items(), key=lambda kv: kv[1]["spearman_max"])
all_sp = [r["spearman"] for v in h1.values() for r in v["runs"]]
report["H1_summary"] = {
    "n_fits": len(all_sp),
    "spearman_overall_min": float(np.min(all_sp)),
    "spearman_overall_max": float(np.max(all_sp)),
    "best_condition": best_overall[0],
    "best_condition_spearman_max": best_overall[1]["spearman_max"],
    "paper_0.82_within_observed_range": bool(np.max(all_sp) >= 0.82),
}
log(f"H1 done: max={np.max(all_sp):.4f} best={best_overall[0]}")

# ---------------------------------------------------------------- H2
# importance는 학습셋만으로 산출해 테스트셋 누수를 피한다.
h2 = {}
for split_kind in ("scaffold", "random"):
    seed = 42
    tr, te = split_indices(split_kind, seed)
    _, model = fit_eval(X_all, tr, te, seed, {"n_estimators": 300}, return_model=True)
    desc_imp = model.feature_importances_[X_fp.shape[1]:]
    order = np.argsort(desc_imp)[::-1]
    ranked = [{"name": desc_names[i], "importance": float(desc_imp[i])} for i in order[:25]]
    h2[f"{split_kind}__top25_descriptors"] = ranked
    log(f"H2 {split_kind} top5: {[d['name'] for d in ranked[:5]]}")

    for k in (0, 10, 25, 50, 100, len(desc_names)):
        X_k = X_fp if k == 0 else np.concatenate([X_fp, X_desc[:, order[:k]]], axis=1)
        r = fit_eval(X_k, tr, te, seed, {"n_estimators": 300})
        h2[f"{split_kind}__k{k}"] = r
        log(f"H2 {split_kind}/k={k}: spearman={r['spearman']:.4f}")
report["H2_descriptor_ablation"] = h2

report["elapsed_sec"] = round(time.time() - T0, 1)
report["note"] = (
    "H1: feature=ECFP4(2048)+RDKit descriptor 전체. seed는 split 셔플과 RF random_state에 동일하게 적용된다. "
    "H2: descriptor importance는 split별 학습셋으로 학습한 RF(n_estimators=300)에서 산출했고, k=0은 fingerprint 단독 대조군이다. "
    "모든 수치는 test set 기준이며 val set은 사용하지 않았다(하이퍼파라미터를 test로 고르지 않도록 전 조합 결과를 그대로 보고한다)."
)

with open("dataset_downloads/reproduction_results_v4_h1_h2.json", "w") as f:
    json.dump(report, f, indent=2)
log("saved reproduction_results_v4_h1_h2.json")
