import pandas as pd, numpy as np, json
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

df = pd.read_csv("dataset_downloads/cycpeptmpdb_all.csv")
df["mol_ok"] = df["smi"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
df = df[df["mol_ok"]].reset_index(drop=True)

report = {}
report["total_rows"] = len(df)
report["n_high_perm_gt_neg6"] = int((df["score"] > -6).sum())
report["n_low_perm_lt_neg6"] = int((df["score"] <= -6).sum())

def sm_to_ecfp4(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = np.stack([sm_to_ecfp4(s) for s in df["smi"]])
y = df["score"].values.astype(float)

def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
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

# LightGBM (as before)
model_lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
model_lgbm.fit(X[train_idx], y[train_idx])
pred_lgbm = model_lgbm.predict(X[test_idx])
report["lightgbm_scaffold_test_r2"] = float(r2_score(y[test_idx], pred_lgbm))
report["lightgbm_scaffold_test_rmse"] = float(root_mean_squared_error(y[test_idx], pred_lgbm))
report["lightgbm_scaffold_test_spearman"] = float(spearmanr(y[test_idx], pred_lgbm).correlation)

# Random Forest (matches paper's actual model choice for permeability)
model_rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model_rf.fit(X[train_idx], y[train_idx])
pred_rf = model_rf.predict(X[test_idx])
report["randomforest_scaffold_test_r2"] = float(r2_score(y[test_idx], pred_rf))
report["randomforest_scaffold_test_rmse"] = float(root_mean_squared_error(y[test_idx], pred_rf))
report["randomforest_scaffold_test_spearman"] = float(spearmanr(y[test_idx], pred_rf).correlation)

# Random split RF+LGBM spearman, for completeness
perm = rng.permutation(n_total)
rtr, rte = perm[:n_train], perm[n_train+n_val:]
model_rf_r = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model_rf_r.fit(X[rtr], y[rtr])
pred_rf_r = model_rf_r.predict(X[rte])
report["randomforest_random_test_spearman"] = float(spearmanr(y[rte], pred_rf_r).correlation)
report["randomforest_random_test_r2"] = float(r2_score(y[rte], pred_rf_r))

report["paper_reported_spearman_RF_fp_desc"] = 0.82
report["note"] = "Paper uses fingerprints+descriptors (not pure ECFP4) and unspecified split method (likely random, not scaffold) for its r=0.82. Our scaffold-split numbers are expected to be lower due to the harder, leakage-free split and fingerprint-only features (no descriptors)."

with open("dataset_downloads/reproduction_results_v2_spearman.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
