"""
evaluate.py — Unified Evaluation Report
=========================================
Combines CNN regression metrics, supervised classifier metrics,
and unsupervised clustering metrics into one clean table.

Sections:
  1. CNN   — MAE, MSE, avg GT count, avg predicted count
  2. XGBoost  — accuracy, weighted F1
  3. KMeans   — silhouette score
  4. GMM      — silhouette score
  5. DBSCAN   — clusters found, anomalies detected
"""

import os
import numpy as np
import scipy.io as sio
import joblib
from tabulate import tabulate
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ─── Paths ────────────────────────────────────────────────────────────────────

DENSITY_DIR = "data/density_maps"
GT_DIR      = "data/part_A_final/test_data/ground_truth"
IMAGE_DIR   = "data/part_A_final/test_data/images"
MODEL_DIR   = "models"

GRID = 8


# ─── Feature extraction (same as other modules) ──────────────────────────────

def extract_patch_features(density_map, grid=GRID):
    """Sync with segment_density.py v2.0 (8 features)."""
    h, w = density_map.shape
    ph, pw = h // grid, w // grid
    features = []
    
    # Needs cv2 for gradient
    import cv2
    
    for i in range(grid):
        for j in range(grid):
            p = density_map[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            m = p.mean()
            s = p.std()
            cv = s / (m + 1e-7)
            
            # Gradient feature
            gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2).max()

            features.append([
                m, p.max(), s, cv, grad_mag,
                (p > m).sum() / (p.size + 1e-7),
                i / (grid - 1), j / (grid - 1),
            ])
    return np.array(features)


# ─── 1. CNN Metrics ──────────────────────────────────────────────────────────

def evaluate_cnn():
    """Compute MAE / MSE between predicted density counts and ground truth."""
    pred_counts, gt_counts = [], []

    for gt_file in sorted(os.listdir(GT_DIR))[:50]:
        mat = sio.loadmat(os.path.join(GT_DIR, gt_file))
        pts = mat["image_info"][0][0][0][0][0]
        gt_cnt = len(pts)

        dm_file = gt_file.replace("GT_IMG_", "IMG_").replace(".mat", ".npy")
        dm_path = os.path.join(DENSITY_DIR, dm_file)

        if not os.path.exists(dm_path):
            continue

        dm = np.load(dm_path)
        pred_cnt = int(dm.sum())

        pred_counts.append(pred_cnt)
        gt_counts.append(gt_cnt)

    mae = mean_absolute_error(gt_counts, pred_counts)
    mse = float(np.mean((np.array(gt_counts) - np.array(pred_counts)) ** 2))

    return {
        "images_evaluated": len(gt_counts),
        "mae": mae,
        "mse": mse,
        "avg_gt": np.mean(gt_counts),
        "avg_pred": np.mean(pred_counts),
    }


# ─── 2. XGBoost Metrics ──────────────────────────────────────────────────────

def evaluate_xgboost(features_scaled):
    """Load saved XGBoost model and evaluate on all patch features."""
    xgb_path = os.path.join(MODEL_DIR, "xgb_classifier.pkl")
    kmeans_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")

    if not os.path.exists(xgb_path) or not os.path.exists(kmeans_path):
        return None

    xgb_model = joblib.load(xgb_path)
    kmeans_model = joblib.load(kmeans_path)

    # Regenerate ground truth from KMeans
    raw_labels = kmeans_model.predict(features_scaled)
    order = np.argsort(kmeans_model.cluster_centers_[:, 0])
    label_map = {order[i]: i for i in range(4)}
    y_true = np.array([label_map[l] for l in raw_labels])

    y_pred = xgb_model.predict(features_scaled)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return {"accuracy": acc, "f1_weighted": f1}


# ─── 3. KMeans Metrics ───────────────────────────────────────────────────────

def evaluate_kmeans(features_scaled):
    """Load saved KMeans model and compute silhouette score."""
    km_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    if not os.path.exists(km_path):
        return None

    kmeans = joblib.load(km_path)
    labels = kmeans.predict(features_scaled)
    sil = silhouette_score(features_scaled, labels)

    return {"silhouette": sil, "k": kmeans.n_clusters}


# ─── 4. GMM Metrics ──────────────────────────────────────────────────────────

def evaluate_gmm(features_scaled):
    """Load saved GMM model and compute silhouette score + confidence."""
    gmm_path = os.path.join(MODEL_DIR, "gmm_model.pkl")
    if not os.path.exists(gmm_path):
        return None

    gmm = joblib.load(gmm_path)
    labels = gmm.predict(features_scaled)
    probs = gmm.predict_proba(features_scaled)
    confidence = probs.max(axis=1).mean()

    sil = silhouette_score(features_scaled, labels)

    return {"silhouette": sil, "avg_confidence": confidence, "n_components": gmm.n_components}


# ─── 5. DBSCAN Metrics ───────────────────────────────────────────────────────

def evaluate_dbscan(features_scaled):
    """Load saved DBSCAN model and report anomaly stats."""
    db_path = os.path.join(MODEL_DIR, "dbscan_model.pkl")
    if not os.path.exists(db_path):
        return None

    dbscan = joblib.load(db_path)
    # DBSCAN doesn't have a predict method for new data; re-fit is needed
    # or we use the stored labels. For consistency, re-run with same params.
    labels = DBSCAN(
        eps=dbscan.eps, min_samples=dbscan.min_samples
    ).fit_predict(features_scaled)

    n_clusters = len(set(labels) - {-1})
    n_anomalies = int((labels == -1).sum())

    return {"clusters": n_clusters, "anomalies": n_anomalies, "total_patches": len(labels)}


# ─── Build unified table ─────────────────────────────────────────────────────

def build_report():
    """
    Load density maps, extract features, evaluate all models,
    and print a single unified table.
    """
    # ── Load features ──
    dm_files = sorted([f for f in os.listdir(DENSITY_DIR) if f.endswith(".npy")])
    all_features = []
    for f in dm_files:
        dm = np.load(os.path.join(DENSITY_DIR, f))
        all_features.append(extract_patch_features(dm))
    all_feats = np.vstack(all_features)

    # Scale features (use saved scaler if available)
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        feats_scaled = scaler.transform(all_feats)
    else:
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(all_feats)

    # ── Evaluate each component ──
    cnn = evaluate_cnn()
    xgb = evaluate_xgboost(feats_scaled)
    km  = evaluate_kmeans(feats_scaled)
    gmm = evaluate_gmm(feats_scaled)
    db  = evaluate_dbscan(feats_scaled)

    # ── Assemble table rows ──
    rows = []

    # Section: CNN
    rows.append(["", "", ""])
    rows.append(["CNN REGRESSION", "", ""])
    rows.append(["  Images evaluated",  cnn["images_evaluated"], ""])
    rows.append(["  MAE",               f'{cnn["mae"]:.2f}',     ""])
    rows.append(["  MSE",               f'{cnn["mse"]:.2f}',     ""])
    rows.append(["  Avg GT count",      f'{cnn["avg_gt"]:.1f}',  ""])
    rows.append(["  Avg predicted",     f'{cnn["avg_pred"]:.1f}',""])

    # Section: XGBoost
    rows.append(["", "", ""])
    rows.append(["SUPERVISED (XGBoost)", "", ""])
    if xgb:
        rows.append(["  Accuracy",      f'{xgb["accuracy"]:.4f}',     ""])
        rows.append(["  F1 (weighted)", f'{xgb["f1_weighted"]:.4f}',  ""])
    else:
        rows.append(["  (model not found)", "—", "Run train_classifier.py first"])

    # Section: KMeans vs GMM
    rows.append(["", "", ""])
    rows.append(["UNSUPERVISED COMPARISON", "", ""])

    if km:
        rows.append(["  KMeans silhouette", f'{km["silhouette"]:.4f}', f'k={km["k"]}'])
    else:
        rows.append(["  KMeans silhouette", "—", "model not found"])

    if gmm:
        rows.append(["  GMM silhouette",    f'{gmm["silhouette"]:.4f}',
                      f'k={gmm["n_components"]}'])
        rows.append(["  GMM avg confidence", f'{gmm["avg_confidence"]:.4f}', ""])
    else:
        rows.append(["  GMM silhouette", "—", "model not found"])

    # Section: DBSCAN
    rows.append(["", "", ""])
    rows.append(["DBSCAN ANOMALY DETECTION", "", ""])
    if db:
        rows.append(["  Clusters found",   db["clusters"],   ""])
        rows.append(["  Anomalies detected", db["anomalies"], f'of {db["total_patches"]} patches'])
        ratio = db["anomalies"] / db["total_patches"] if db["total_patches"] > 0 else 0
        rows.append(["  Anomaly ratio",    f"{ratio:.4f}",   ""])
    else:
        rows.append(["  (model not found)", "—", "Run segment_density.py first"])

    # ── Print ──
    print("\n" + "=" * 70)
    print("  UNIFIED EVALUATION REPORT")
    print("=" * 70)
    print(tabulate(rows, headers=["Metric", "Value", "Notes"], tablefmt="grid"))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_report()
