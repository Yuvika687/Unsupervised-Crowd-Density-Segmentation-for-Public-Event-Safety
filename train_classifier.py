"""
train_classifier.py — Supervised + Unsupervised Patch Classifier
=================================================================
1. Extracts 6 features per 8x8 patch from predicted density maps.
2. Uses KMeans-derived labels (Low=0 / Medium=1 / High=2 / Critical=3)
   as ground truth for an XGBoost supervised classifier.
3. Also fits a GaussianMixture (n_components=4) as an unsupervised
   alternative.

Outputs:
  models/xgb_classifier.pkl   — trained XGBoost model
  models/gmm_model.pkl        — trained GMM model
  models/feature_scaler.pkl   — StandardScaler used for features
  models/kmeans_model.pkl     — KMeans used for label generation
"""

import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

DENSITY_MAP_DIR = "data/density_maps"
MODEL_DIR       = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
GRID = 8


def extract_patch_features(density_map, grid=GRID):
    """
    Standardizes feature extraction to 8 features.
    Matches segment_density.py and app.py.
    """
    h, w   = density_map.shape
    ph, pw = h // grid, w // grid
    features = []
    for i in range(grid):
        for j in range(grid):
            patch = density_map[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            
            # Base stats
            m = patch.mean()
            s = patch.std()
            
            # Derived metrics for complex crowd structures
            cv = s / (m + 1e-7)
            
            # Sobel gradients for texture density
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


def load_all_features(density_dir):
    files       = sorted([f for f in os.listdir(density_dir)
                          if f.endswith(".npy")])
    all_features = []
    for f in files:
        dm    = np.load(os.path.join(density_dir, f))
        feats = extract_patch_features(dm)
        all_features.append(feats)
    return np.vstack(all_features)


def generate_labels(features_scaled, n_clusters=4):
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(features_scaled)
    order      = np.argsort(kmeans.cluster_centers_[:, 0])
    label_map  = {order[i]: i for i in range(n_clusters)}
    labels     = np.array([label_map[l] for l in raw_labels])
    return labels, kmeans


def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)

    label_names   = ["Low", "Medium", "High", "Critical"]
    unique_labels = sorted(set(y_test))
    names_present = [label_names[i] for i in unique_labels]

    print("\n" + "=" * 60)
    print("  XGBoost Classifier Results")
    print("=" * 60)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (wt.) : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=names_present))

    return clf, acc, f1


def train_gmm(X, n_components=4):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )
    gmm.fit(X)

    probs  = gmm.predict_proba(X)
    labels = gmm.predict(X)

    print("\n" + "=" * 60)
    print("  GMM (Unsupervised) Results")
    print("=" * 60)
    print(f"  Components : {n_components}")
    print(f"  BIC        : {gmm.bic(X):.2f}")
    print(f"  AIC        : {gmm.aic(X):.2f}")
    print(f"  Avg confidence : {probs.max(axis=1).mean():.4f}")
    print(f"  Label dist : "
          f"{ dict(zip(*np.unique(labels, return_counts=True))) }")

    return gmm


def main():
    print("Loading density maps and extracting features ...")
    features = load_all_features(DENSITY_MAP_DIR)
    print(f"  Total patches : {features.shape[0]}  |  "
          f"Features/patch : {features.shape[1]}")

    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("\nGenerating KMeans labels ...")
    labels, kmeans  = generate_labels(features_scaled)
    label_names     = ["Low", "Medium", "High", "Critical"]
    print(f"  Label distribution : "
          f"{ {label_names[i]: int((labels==i).sum()) for i in range(4)} }")

    print("\nTraining XGBoost classifier ...")
    xgb_model, xgb_acc, xgb_f1 = train_xgboost(features_scaled, labels)

    print("\nTraining GMM ...")
    gmm_model = train_gmm(features_scaled)

    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_classifier.pkl"))
    joblib.dump(gmm_model, os.path.join(MODEL_DIR, "gmm_model.pkl"))
    joblib.dump(scaler,    os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    joblib.dump(kmeans,    os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    print("\n" + "=" * 60)
    print("  All models saved to models/")
    print("=" * 60)
    print("  models/xgb_classifier.pkl")
    print("  models/gmm_model.pkl")
    print("  models/feature_scaler.pkl")
    print("  models/kmeans_model.pkl")


if __name__ == "__main__":
    main()