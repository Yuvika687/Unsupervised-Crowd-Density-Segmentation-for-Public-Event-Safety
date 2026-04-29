"""
segment_density.py — Multi-Method Crowd Density Segmentation
==============================================================
1. KMeans clustering  (k=4)          — original baseline
2. DBSCAN anomaly detection          — flags density outliers
3. GMM soft clustering               — posterior confidence scores

All methods operate on 6 features extracted from 8×8 patch grids
of predicted density maps.

Outputs saved to results/ and models/.
"""

import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# ─── Paths ────────────────────────────────────────────────────────────────────

DENSITY_MAP_DIR = "data/density_maps"
IMAGE_DIR       = "data/part_A_final/test_data/images"
OUTPUT_DIR      = "results/segmentation"
MODEL_DIR       = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GRID = 8


# ─── Feature extraction ──────────────────────────────────────────────────────

def extract_patch_features(density_map, grid=GRID):
    """
    Split density map into grid×grid patches and compute 8 features each:
      1. mean density
      2. max density
      3. std of density
      4. coefficient of variation (std/mean) -> cluster clumpiness
      5. max gradient (edge intensity)       -> crowd structure
      6. fraction of pixels above patch mean
      7. normalised row position  (0 → 1)
      8. normalised col position  (0 → 1)
    """
    h, w = density_map.shape
    ph, pw = h // grid, w // grid
    features = []

    for i in range(grid):
        for j in range(grid):
            patch = density_map[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            
            # Feature calculation with safety for division by zero
            m = patch.mean()
            s = patch.std()
            cv = s / (m + 1e-7)
            
            # Gradient feature
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2).max()

            features.append([
                m,
                patch.max(),
                s,
                cv,
                grad_mag,
                (patch > m).sum() / (patch.size + 1e-7),
                i / (grid - 1),
                j / (grid - 1),
            ])

    return np.array(features)


def load_density_maps(density_dir):
    """Load all .npy density maps from directory."""
    files = sorted([f for f in os.listdir(density_dir) if f.endswith(".npy")])
    maps, names = [], []
    for f in files:
        dm = np.load(os.path.join(density_dir, f))
        maps.append(dm)
        names.append(f)
    return maps, names


# ─── 1. KMeans Clustering ────────────────────────────────────────────────────

def segment_kmeans(features_scaled, n_clusters=4):
    """Run KMeans(k=4), sort clusters by mean density, assign safety labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    sil = silhouette_score(features_scaled, labels)

    # Sort clusters by mean density (feature index 0) → ordered labels
    order = np.argsort(kmeans.cluster_centers_[:, 0])
    label_names_map = {order[0]: "Low", order[1]: "Medium",
                       order[2]: "High", order[3]: "Critical"}
    named = [label_names_map[l] for l in labels]

    print("=" * 60)
    print("  KMeans Clustering (k=4)")
    print("=" * 60)
    print(f"  Silhouette Score : {sil:.4f}")
    print(f"  Label distribution : { {l: named.count(l) for l in ['Low', 'Medium', 'High', 'Critical']} }")

    return labels, named, kmeans, sil


# ─── 2. DBSCAN Anomaly Detection ─────────────────────────────────────────────

def segment_dbscan(features_scaled, eps=0.8, min_samples=5):
    """
    Run DBSCAN for anomaly detection.
    Points labelled -1 are anomalies (density outliers).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)

    n_clusters = len(set(labels) - {-1})
    n_anomalies = (labels == -1).sum()

    print("\n" + "=" * 60)
    print("  DBSCAN Anomaly Detection")
    print("=" * 60)
    print(f"  eps={eps}, min_samples={min_samples}")
    print(f"  Clusters found    : {n_clusters}")
    print(f"  Anomalies (noise) : {n_anomalies}")
    print(f"  Anomaly ratio     : {n_anomalies / len(labels):.4f}")

    return labels, dbscan, n_clusters, n_anomalies


# ─── 3. GMM Soft Clustering ──────────────────────────────────────────────────

def segment_gmm(features_scaled, n_components=4):
    """
    Fit a Gaussian Mixture Model for soft clustering.
    Returns cluster labels AND posterior probability (confidence) per patch.
    """
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    gmm.fit(features_scaled)

    labels = gmm.predict(features_scaled)
    probs = gmm.predict_proba(features_scaled)  # (n_samples, n_components)
    confidence = probs.max(axis=1)  # max posterior = assignment confidence

    sil = silhouette_score(features_scaled, labels)

    print("\n" + "=" * 60)
    print("  GMM Soft Clustering (k=4)")
    print("=" * 60)
    print(f"  Silhouette Score      : {sil:.4f}")
    print(f"  BIC                   : {gmm.bic(features_scaled):.2f}")
    print(f"  AIC                   : {gmm.aic(features_scaled):.2f}")
    print(f"  Mean confidence       : {confidence.mean():.4f}")
    print(f"  Min  confidence       : {confidence.min():.4f}")
    print(f"  Label distribution    : {dict(zip(*np.unique(labels, return_counts=True)))}")

    return labels, gmm, sil, confidence


# ─── Main execution ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load data ──
    density_maps, names = load_density_maps(DENSITY_MAP_DIR)
    print(f"Loaded {len(density_maps)} density maps.\n")

    # ── Extract & scale features ──
    all_features = []
    for dm in density_maps:
        feats = extract_patch_features(dm)
        all_features.append(feats)

    all_feats_flat = np.vstack(all_features)
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(all_feats_flat)

    print(f"Total patches : {feats_scaled.shape[0]}  |  Features/patch : {feats_scaled.shape[1]}\n")

    # ── 1. KMeans ──
    km_labels, km_named, kmeans_model, km_sil = segment_kmeans(feats_scaled)

    # ── 2. DBSCAN ──
    db_labels, dbscan_model, db_n_clusters, db_n_anomalies = segment_dbscan(feats_scaled)

    # ── 3. GMM ──
    gmm_labels, gmm_model, gmm_sil, gmm_conf = segment_gmm(feats_scaled)

    # ── Save models ──
    joblib.dump(kmeans_model, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(dbscan_model, os.path.join(MODEL_DIR, "dbscan_model.pkl"))
    joblib.dump(gmm_model,    os.path.join(MODEL_DIR, "gmm_model.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "feature_scaler.pkl"))

    # ── Save results ──
    np.save("results/patch_labels_kmeans.npy", np.array(km_named))
    np.save("results/patch_labels_dbscan.npy", db_labels)
    np.save("results/patch_labels_gmm.npy", gmm_labels)
    np.save("results/gmm_confidence.npy", gmm_conf)

    print("\n" + "=" * 60)
    print("  All models & results saved")
    print("=" * 60)
    print("  Models  → models/kmeans_model.pkl, dbscan_model.pkl, gmm_model.pkl")
    print("  Results → results/patch_labels_*.npy, gmm_confidence.npy")
