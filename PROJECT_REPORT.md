# Unsupervised Crowd Density Segmentation for Public Event Safety

## 1. Project Overview

This project builds a crowd-safety analysis system for public events. The system takes a crowd image, estimates the number of people, converts the scene into density zones, identifies risky areas, and presents the output in a Streamlit dashboard for safety monitoring.

The main goal is not only crowd counting, but also crowd-risk interpretation. Instead of giving only a final count, the project divides the image into local zones and labels them as `Low`, `Medium`, `High`, or `Critical` risk so event organizers can react faster.

## 2. Problem Statement

Large public gatherings such as concerts, festivals, rallies, and sports events can become unsafe when crowd density rises in a few local areas. Manual monitoring is slow and subjective. The project solves this by using computer vision and machine learning to:

- estimate crowd count from a single image
- generate density maps
- classify local crowd-risk zones
- highlight dangerous regions visually
- support quick safety decisions for crowd control

## 3. Objectives

- Build an automated crowd counting pipeline.
- Detect dense and risky local zones in the scene.
- Compare unsupervised and supervised segmentation methods.
- Provide an interactive dashboard for live analysis.
- Support public-event safety use cases such as monitoring, alerts, and evacuation planning.

## 4. Main System Architecture

The project has five major layers:

1. Data preparation
2. Crowd counting model
3. Patch-level density segmentation
4. Safety analytics and threat scoring
5. User interface and decision support

### End-to-end flow

`Input image -> density estimation -> 8x8 patch division -> feature extraction -> zone classification -> threat scoring -> visual dashboard`

## 5. Repository Components

### Main application

- `app.py`: full Streamlit dashboard and integrated analysis system

### Training and inference

- `training/train.py`: older baseline counting pipeline
- `training/train_v2.py`: improved dual-head training pipeline
- `training/model_v2.py`: V2 architecture
- `training/inference_v2.py`: tiled inference with test-time augmentation
- `training/calibrate_v2.py`: raw-to-count calibration for V2
- `training/eval_v2.py`: V2 evaluation
- `training/eval_lwcc.py`: evaluation for LWCC DM-Count

### Segmentation and classification

- `training/segment_density.py`: KMeans, DBSCAN, GMM segmentation
- `training/train_classifier.py`: XGBoost classifier trained from KMeans labels
- `training/evaluate.py`: unified evaluation of classifiers and clustering models

### Dataset processing

- `scripts/preprocess_dataset.py`: density-map generation for ShanghaiTech Part A
- `scripts/dataset_v2.py`: adaptive density maps, multi-dataset loading, sampling, augmentation

## 6. Datasets Used

The repository includes or references the following datasets:

- ShanghaiTech Part A
  - Train images: 300
  - Test images: 182
  - Dense crowd scenes

- ShanghaiTech Part B
  - Train images: 400
  - Test images: 316
  - Sparse-to-medium crowd scenes

- JHU Crowd V2.0
  - Train images: 2272
  - Val images: 500
  - Test images: 1600
  - In the V2 pipeline, only images with count `<= 100` are filtered for sparse-scene learning
  - Filtered counts:
    - Train kept: 1076
    - Val kept: 258
    - Test kept: 737

- UCSD and Mall datasets are also referenced in the older baseline pipeline.

## 7. Methodology

### 7.1 Data preprocessing

The project converts ground-truth head annotations into density maps.

- `scripts/preprocess_dataset.py` uses Gaussian smoothing on annotated head points.
- `scripts/dataset_v2.py` improves this with geometry-adaptive Gaussian kernels using KDTree neighbor distances.
- Adaptive sigma helps represent both sparse and dense scenes more realistically.

### 7.2 Crowd counting models

#### A. Legacy baseline model

The older pipeline in `training/train.py` uses:

- VGG16 encoder
- transposed-convolution decoder
- density-map regression

This version combines ShanghaiTech with additional sparse datasets, but some parts of that legacy script still use placeholder point generation for UCSD and Mall. So it should be treated as an experimental baseline, not the strongest final system.

#### B. V2 custom model

The improved model in `training/model_v2.py` is a dual-head architecture:

- encoder: VGG16 feature extractor
- head 1: density decoder
- head 2: direct count regressor
- final fused count:
  - `0.7 * density_sum + 0.3 * direct_count`

Training improvements in `training/train_v2.py` include:

- focal-weighted MSE
- SSIM loss for structure preservation
- log-cosh count loss
- EMA validation model
- warmup + cosine learning-rate schedule
- gradient accumulation
- early stopping
- multi-scale crops and augmentation

#### C. LWCC DM-Count path

The deployed app primarily uses LWCC DM-Count when available.

- SHB weights are used first for sparse scenes
- SHA weights are used for denser scenes
- this adaptive strategy is implemented in `app.py` and `training/eval_lwcc.py`

This is currently the strongest counting path in the repository.

### 7.3 Zone-based density segmentation

After density estimation, each scene is split into an `8x8` grid, giving `64` local patches.

For every patch, the system extracts 8 features:

- mean density
- max density
- standard deviation
- coefficient of variation
- max gradient magnitude
- fraction of pixels above patch mean
- normalized row position
- normalized column position

These features are then used by different segmentation methods:

- `KMeans`
  - unsupervised hard clustering into 4 groups
  - clusters are ordered as `Low`, `Medium`, `High`, `Critical`

- `DBSCAN`
  - detects unusual density patches as anomalies

- `GMM`
  - soft clustering with probability/confidence

- `XGBoost`
  - supervised classifier trained on KMeans-generated pseudo-labels

### 7.4 Safety analytics

The system does more than count people.

It also computes:

- zone-wise risk map
- total crowd count
- threat score
- threat label
- critical/high/medium/low zone counts
- marker-based head visualization
- portrait fallback using face detection for close-up scenes

### 7.5 Dashboard and interface

The Streamlit app in `app.py` contains these main tabs:

- `Live Analysis`
- `Compare Methods`
- `Dashboard`
- `Live Capture`
- `Batch Analysis`
- `RL Agent`

Important app capabilities:

- image upload and instant crowd analysis
- density heatmap overlay
- zone-risk overlay
- head-dot visualization
- side-by-side comparison of KMeans, XGBoost, and GMM
- mobile live capture through phone QR link
- batch image analysis
- DQN-based evacuation recommendation demo

## 8. Key Results From Current Repository State

### 8.1 LWCC DM-Count evaluation

Using `results/eval_lwcc_detail.csv`:

- Total evaluated images: 498
- Overall MAE: 45.36
- Overall RMSE: 89.25
- Overall bias: -26.90
- Overall estimated accuracy: 80.89%

For sparse scenes with ground-truth count `1-100`:

- Images: 176
- MAE: 5.80
- RMSE: 9.26
- Bias: -4.26
- Accuracy: 90.25%

Per dataset:

- Part B MAE: 22.35
- Part A MAE: 85.32

Interpretation:

- LWCC performs well on sparse and medium crowd scenes.
- Performance drops on very dense scenes, but it is still much better than the custom V2 branch at the moment.

### 8.2 V2 custom model evaluation

Using `results/eval_v2_detail.csv`:

- Total evaluated images: 798
- Overall MAE: 159.55
- Overall RMSE: 279.10
- Overall bias: -159.36
- Overall estimated accuracy: 3.43%

Sparse scenes with ground-truth count `1-100`:

- Images: 476
- MAE: 48.04
- RMSE: 54.02
- Bias: -47.73
- Accuracy: 5.28%

Interpretation:

- The V2 branch is currently undercounting heavily.
- It is useful as a research/improvement branch, but not the best final counting backend for presentation as the main result.

### 8.3 Patch classification and segmentation results

Using `training/evaluate.py` with saved models:

- XGBoost accuracy: 0.9986
- XGBoost weighted F1: 0.9986
- KMeans silhouette score: 0.3119
- GMM silhouette score: 0.2210
- GMM average confidence: 0.9835
- DBSCAN clusters found: 8
- DBSCAN anomalies detected: 712 of 19200 patches
- DBSCAN anomaly ratio: 0.0371

Interpretation:

- XGBoost is very strong at reproducing the KMeans-derived zone labels.
- KMeans and GMM are useful for unsupervised interpretation.
- DBSCAN helps identify abnormal local crowd patterns.

## 9. Strengths of the Project

- Combines crowd counting with safety-zone interpretation.
- Supports both unsupervised and supervised zone classification.
- Includes an interactive dashboard instead of only offline scripts.
- Handles sparse scenes well with LWCC DM-Count.
- Provides visual overlays that are easy for non-technical users to understand.
- Includes mobile capture and batch analysis features for real deployment demos.

## 10. Limitations

- The custom V2 model is still underperforming and undercounting.
- Dense-scene accuracy is weaker than sparse-scene accuracy.
- The older `training/train.py` pipeline contains placeholder logic for some auxiliary datasets.
- The unified evaluation script for the legacy CNN path uses stored density maps rather than a strong final end-to-end inference benchmark.
- The RL agent is a demonstration component for evacuation policy learning, not a validated real-world evacuation controller.

## 11. Practical Use Cases

- monitoring crowd density during festivals or concerts
- identifying local stampede-risk zones
- supporting security teams with visual safety alerts
- assisting venue operators in controlling entry, movement, and evacuation
- generating explainable analytics for public-event management

## 12. Conclusion

This project presents a strong idea for public-event safety: combine crowd counting with local zone-based safety segmentation. The best practical result in the current repository is the LWCC DM-Count counting pipeline, especially for sparse and medium-density scenes. The zone-classification layer built with KMeans, XGBoost, GMM, and DBSCAN adds meaningful safety interpretation beyond a simple count.

So the project’s main contribution is not only counting people, but converting crowd density into actionable public-safety intelligence.

## 13. Equal Work Division for 2 Team Members

Below is a balanced split so both members contribute equally in technical work and presentation work.

### Member 1: Data + Crowd Counting Pipeline

Responsibilities:

- dataset collection and organization
- annotation understanding for ShanghaiTech and JHU
- density-map generation
- preprocessing scripts
- study of baseline CNN and V2 model
- LWCC DM-Count integration and evaluation
- count-based performance analysis

Expected deliverables:

- dataset section of report
- preprocessing workflow
- counting-model explanation
- evaluation tables for MAE, RMSE, bias, accuracy

### Member 2: Segmentation + Safety Dashboard

Responsibilities:

- patch feature extraction
- KMeans, GMM, DBSCAN implementation
- XGBoost zone classifier
- safety-zone labeling logic
- Streamlit dashboard integration
- mobile capture, batch analysis, and RL demo explanation
- visualization and UI screenshots

Expected deliverables:

- methodology section for zone classification
- dashboard/demo section
- segmentation result analysis
- application and use-case explanation

### Why this split is equal

Member 1 handles the full counting pipeline and quantitative evaluation.

Member 2 handles the full safety analytics, classification pipeline, visualization, and user interaction layer.

Both sides are comparable in difficulty:

- Member 1 covers data engineering + deep learning + evaluation
- Member 2 covers ML segmentation + product integration + decision-support logic

## 14. Equal Presentation Split for 2 Members

If your presentation is around 10 minutes, use this split:

### Member 1 speaking plan: first 5 minutes

- title and problem statement
- motivation and objectives
- datasets used
- preprocessing and density-map generation
- crowd counting models
- LWCC and V2 result comparison

### Member 2 speaking plan: next 5 minutes

- patch segmentation approach
- KMeans, XGBoost, GMM, DBSCAN
- safety-zone overlays and threat scoring
- Streamlit dashboard features
- mobile capture and batch analysis
- conclusion, limitations, and future work

## 15. Short Future Work Section

- improve dense-scene accuracy
- replace remaining legacy placeholder dataset logic
- retrain the custom V2 model with stronger calibration and validation
- test on video streams instead of only still images
- connect RL recommendations with a more realistic crowd simulator

## 16. One-Line Project Summary for Presentation

This project builds an intelligent crowd-monitoring system that estimates crowd size, detects risky dense zones, and displays real-time safety insights for public-event management.
