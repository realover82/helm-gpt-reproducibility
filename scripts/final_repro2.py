import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import json

base = "dataset_downloads/cpp_authors"
data_df = pd.read_csv("dataset_downloads/cycpeptmpdb_all.csv")

X_fps = np.load(f"{base}/X_fps.npy.try")
X_dps = np.load(f"{base}/X_dps.npy.try")
X = np.concatenate([X_fps, X_dps], axis=1)
y = data_df["score"].values

with open(f"{base}/scaffold_kfold.pkl", "rb") as f:
    split_idx = pickle.load(f)

result = {}
result["X_fps_shape"] = list(X_fps.shape)
result["X_dps_shape"] = list(X_dps.shape)

def run_fold(val_split):
    val_idx = split_idx[val_split]
    test_idx = split_idx[val_split + 1]
    train_splits = [split_idx[i] for i in range(len(split_idx)) if i != val_split + 1 and i != val_split]
    train_idx = np.concatenate(train_splits, axis=0)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_val_hat = model.predict(X_val)
    y_test_hat = model.predict(X_test)
    sp_val = stats.spearmanr(y_val_hat, y_val).correlation
    sp_test = stats.spearmanr(y_test_hat, y_test).correlation
    y_hat_vt = np.concatenate([y_val_hat, y_test_hat])
    y_vt = np.concatenate([y_val, y_test])
    sp_vt = stats.spearmanr(y_hat_vt, y_vt).correlation
    return sp_val, sp_test, sp_vt

# Primary: authors' exact val_split=7 (their reported fold)
sp_val, sp_test, sp_vt = run_fold(7)
result["authors_exact_split_val_spearman"] = float(sp_val)
result["authors_exact_split_test_spearman"] = float(sp_test)
result["authors_exact_split_val_plus_test_spearman"] = float(sp_vt)

# Full cross-validation across all valid fold pairs, to directly compare to paper's Fig 2a (0.547 +/- 0.142)
all_test_sp = []
n_folds = len(split_idx)
for vs in range(0, n_folds - 1):
    try:
        _, sp_t, _ = run_fold(vs)
        all_test_sp.append(sp_t)
    except Exception as e:
        all_test_sp.append(None)

result["cv_test_spearman_per_fold"] = all_test_sp
valid = [s for s in all_test_sp if s is not None]
result["cv_test_spearman_mean"] = float(np.mean(valid))
result["cv_test_spearman_std"] = float(np.std(valid))
result["paper_fig2a_RF_FPS_DPS_mean_std"] = [0.547, 0.142]
result["paper_fig2c_final_model_reported"] = 0.82
result["note"] = ("Uses the AUTHORS' OWN precomputed fps(2048)+descriptor(211) feature matrices and their exact "
                   "scaffold_k_fold split indices (data/cpp/scaffold_k_fold_idxes.pkl), with a freshly trained "
                   "RandomForestRegressor(n_estimators=300, random_state=42) since their saved pickle is sklearn "
                   "0.23.2-incompatible with current sklearn. This isolates the comparison to just "
                   "RF-hyperparameter/random_state differences from the authors' own (unknown) RF config - "
                   "everything else (features, split) is now identical to the authors' pipeline.")

print(json.dumps(result, indent=2))
with open("dataset_downloads/final_authors_data_repro.json", "w") as f:
    json.dump(result, f, indent=2)
