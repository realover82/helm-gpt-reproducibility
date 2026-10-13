import pandas as pd, numpy as np, json
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import spearmanr

df = pd.read_csv("dataset_downloads/cycpeptmpdb_all.csv")
df["mol_ok"] = df["smi"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
df = df[df["mol_ok"]].reset_index(drop=True)

report = {}
report["total_rows"] = len(df)

def sm_to_ecfp4(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

desc_names = [n for n, f in Descriptors._descList]
report["n_rdkit_descriptors"] = len(desc_names)

def sm_to_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    vals = []
    for name, func in Descriptors._descList:
        try:
            v = func(mol)
        except Exception:
            v = np.nan
        vals.append(v)
    return vals

X_fp = np.stack([sm_to_ecfp4(s) for s in df["smi"]])
X_desc = np.array([sm_to_desc(s) for s in df["smi"]])
X_desc = np.nan_to_num(X_desc, nan=0.0, posinf=0.0, neginf=0.0)
X_combined = np.concatenate([X_fp, X_desc], axis=1)
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

perm = rng.permutation(n_total)
rtr, rte = perm[:n_train], perm[n_train+n_val:]

def run(X, train_i, test_i, label):
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X[train_i], y[train_i])
    pred = model.predict(X[test_i])
    report[label + "_spearman"] = float(spearmanr(y[test_i], pred).correlation)
    report[label + "_r2"] = float(r2_score(y[test_i], pred))
    report[label + "_rmse"] = float(root_mean_squared_error(y[test_i], pred))

run(X_fp, train_idx, test_idx, "fp_only_scaffold")
run(X_combined, train_idx, test_idx, "fp_plus_descriptors_scaffold")
run(X_fp, rtr, rte, "fp_only_random")
run(X_combined, rtr, rte, "fp_plus_descriptors_random")

report["paper_reported_spearman"] = 0.82
report["note"] = "fp_plus_descriptors uses full RDKit Descriptors._descList (210 descriptors) concatenated with ECFP4(2048-bit), matching the paper's stated feature combination (fingerprints + descriptors e.g. EState_VSA, MolLogP). Comparison to fp_only (v2 result) isolates the descriptor contribution."

with open("dataset_downloads/reproduction_results_v3_descriptors.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
