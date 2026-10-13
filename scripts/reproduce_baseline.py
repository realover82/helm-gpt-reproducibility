import pandas as pd, numpy as np, json, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

report = {}

# ---------- 1. CycPeptMPDB: real regression baseline ----------
df = pd.read_csv("dataset_downloads/cycpeptmpdb_all.csv")
report["cycpeptmpdb_raw_rows"] = len(df)

def sm_to_ecfp4(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

df["mol_ok"] = df["smi"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
report["cycpeptmpdb_valid_smiles"] = int(df["mol_ok"].sum())
report["cycpeptmpdb_invalid_smiles"] = int((~df["mol_ok"]).sum())
df = df[df["mol_ok"]].reset_index(drop=True)

X = np.stack([sm_to_ecfp4(s) for s in df["smi"]])
y = df["score"].values.astype(float)
report["target_col"] = "score (-logPapp membrane permeability)"
report["target_min"] = float(y.min())
report["target_max"] = float(y.max())
report["target_mean"] = float(y.mean())

def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

scaffold_to_idx = defaultdict(list)
for idx, smi in enumerate(df["smi"]):
    scaffold_to_idx[generate_scaffold(smi)].append(idx)

clusters = list(scaffold_to_idx.values())
rng = np.random.RandomState(42)
rng.shuffle(clusters)
clusters.sort(key=len, reverse=True)

n_total = len(df)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
train_idx, val_idx, test_idx = [], [], []
for c in clusters:
    if len(train_idx) + len(c) <= n_train:
        train_idx.extend(c)
    elif len(val_idx) + len(c) <= n_val:
        val_idx.extend(c)
    else:
        test_idx.extend(c)

report["n_scaffold_clusters"] = len(clusters)
report["split_sizes"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}

model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
model.fit(X[train_idx], y[train_idx])

pred_val = model.predict(X[val_idx])
pred_test = model.predict(X[test_idx])

report["val_rmse"] = float(root_mean_squared_error(y[val_idx], pred_val))
report["val_r2"] = float(r2_score(y[val_idx], pred_val))
report["test_rmse"] = float(root_mean_squared_error(y[test_idx], pred_test))
report["test_mae"] = float(mean_absolute_error(y[test_idx], pred_test))
report["test_r2"] = float(r2_score(y[test_idx], pred_test))

perm = rng.permutation(n_total)
rtr, rva, rte = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]
model_r = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
model_r.fit(X[rtr], y[rtr])
pred_rte = model_r.predict(X[rte])
report["random_split_test_r2"] = float(r2_score(y[rte], pred_rte))
report["random_split_test_rmse"] = float(root_mean_squared_error(y[rte], pred_rte))

# ---------- 2. ChEMBL HELM corpus: EDA only (no regression target present) ----------
chembl = pd.read_csv("dataset_downloads/chembl_helm_subset.csv")
report["chembl_raw_rows"] = len(chembl)
report["chembl_columns"] = list(chembl.columns)
chembl["mol_ok"] = chembl["cano_smi"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None if pd.notna(s) else False)
report["chembl_valid_smiles"] = int(chembl["mol_ok"].sum())
report["chembl_invalid_smiles"] = int((~chembl["mol_ok"]).sum())
report["chembl_helm_len_mean"] = float(chembl["helm_len"].mean())
report["chembl_helm_len_median"] = float(chembl["helm_len"].median())
report["chembl_has_activity_label"] = False
report["chembl_note"] = "No IC50/Ki/bioactivity column in this file - it is a HELM sequence corpus for GPT prior pretraining, not a supervised regression dataset. Supervised reproduction was not possible on this file; only data-quality EDA performed."

with open("dataset_downloads/reproduction_results.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
