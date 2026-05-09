# Detailed Project Report

## Project Title

**Unsupervised Crowd Density Segmentation for Public Event Safety**

## Team Details

- Team size: 2 members
- Suggested member placeholders:
  - Member 1: __________________
  - Member 2: __________________
- Repository base: local project implementation in this workspace

## 1. Abstract

This project develops an intelligent crowd-analysis system for public-event safety. The system accepts a crowd image, estimates the number of people, generates a density map, divides the scene into local regions, and classifies those regions into safety levels such as `Low`, `Medium`, `High`, and `Critical`. The final output is presented in a Streamlit dashboard that supports live analysis, visual overlays, batch processing, mobile image capture, and a reinforcement-learning-based evacuation demo.

The project combines multiple approaches instead of relying on a single model. For crowd counting, the repository contains both a custom CNN-based pipeline and an LWCC DM-Count integration. For zone understanding, it uses KMeans, GMM, DBSCAN, and an XGBoost classifier trained on pseudo-labels. The current repository state shows that the LWCC DM-Count path is the strongest practical counting backend, especially for sparse and medium-density scenes, while the zone-classification stack provides actionable safety insight beyond raw person count.

## 2. Introduction

Crowd management is one of the most important safety challenges in large public gatherings such as concerts, rallies, festivals, sports events, religious gatherings, and political programs. A crowd can become dangerous even when the overall number of people is not extreme, because the actual risk often appears in localized dense pockets rather than uniformly across the full venue.

Traditional crowd monitoring depends heavily on human observation through CCTV feeds. That approach is limited by fatigue, subjectivity, and delayed reaction. A computer-vision-based monitoring system can assist security teams by continuously estimating crowd density and highlighting high-risk zones automatically.

This project addresses that need by building a system that does not stop at crowd counting. It adds patch-wise density segmentation and safety labeling so the output is more useful for practical decision-making.

## 3. Problem Statement

The problem addressed in this project is:

**How can we automatically analyze crowd images and convert them into useful safety intelligence for public-event monitoring?**

The system must answer three practical questions:

1. How many people are present?
2. Which local areas are becoming overcrowded?
3. How risky is the current scene from a safety perspective?

## 4. Motivation

The project is motivated by the following real-world requirements:

- Public events require rapid detection of overcrowded regions.
- Security teams need visual, explainable outputs instead of only numbers.
- Manual counting is unreliable in dense scenes.
- Early warnings can reduce the chance of stampede-like situations.
- A usable dashboard is necessary for deployment and demonstration.

## 5. Objectives

- Build a crowd counting system using density estimation.
- Generate density maps from head annotations.
- Divide the scene into local zones and analyze patch-level density.
- Compare unsupervised and supervised methods for zone classification.
- Create a safety-aware dashboard for analysis and visualization.
- Demonstrate how crowd intelligence can support public-event safety.

## 6. Scope of the Project

This project focuses on image-based crowd analysis and safety interpretation. It covers:

- dataset preparation
- density-map generation
- deep-learning-based crowd counting
- unsupervised clustering of local density patterns
- supervised zone classification
- safety visualization
- dashboard-based interaction

The current implementation is image-centric. It is not a production-grade end-to-end surveillance platform and does not yet provide a fully validated evacuation control system.

## 7. Repository Overview

The repository is organized around application, training, data preparation, and evaluation components.

### Main files and folders

| Path | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `training/train.py` | Older baseline crowd-counting training pipeline |
| `training/train_v2.py` | Improved V2 training pipeline |
| `training/model_v2.py` | Dual-head V2 model |
| `training/inference_v2.py` | V2 tiled inference with TTA |
| `training/calibrate.py` | Calibration for baseline model |
| `training/calibrate_v2.py` | Calibration for V2 model |
| `training/eval_lwcc.py` | LWCC DM-Count evaluation |
| `training/eval_v2.py` | V2 model evaluation |
| `training/segment_density.py` | KMeans, GMM, DBSCAN segmentation |
| `training/train_classifier.py` | XGBoost patch classifier |
| `training/evaluate.py` | Unified evaluation report |
| `scripts/preprocess_dataset.py` | Density-map generation |
| `scripts/dataset_v2.py` | Multi-dataset adaptive loader |
| `models/` | Saved weights and ML artifacts |
| `results/` | Evaluation outputs and visual artifacts |

## 8. Datasets Used

The repository contains data references for multiple crowd datasets.

### 8.1 ShanghaiTech Part A

- Training images: 300
- Test images: 182
- Typical characteristic: dense crowd scenes
- Use in project:
  - density-map generation
  - baseline and V2 training
  - dense-scene evaluation

### 8.2 ShanghaiTech Part B

- Training images: 400
- Test images: 316
- Typical characteristic: sparse to medium crowd scenes
- Use in project:
  - counting evaluation
  - sparse-scene validation
  - app presentation metrics

### 8.3 JHU Crowd V2.0

- Training images: 2272
- Validation images: 500
- Test images: 1600
- V2 pipeline uses filtered images with crowd count `<= 100` for sparse-scene training

Filtered counts observed in this repository:

| Split | Total | Kept with count `<= 100` |
|---|---:|---:|
| Train | 2272 | 1076 |
| Val | 500 | 258 |
| Test | 1600 | 737 |

### 8.4 UCSD and Mall

These appear in the older baseline training and calibration scripts.

- UCSD is treated as a small-crowd dataset.
- Mall is treated as a small-to-medium crowd dataset.

Important note:

The older baseline script uses simplified or placeholder logic for some auxiliary datasets, so those parts should be presented carefully as supportive experimentation rather than the most rigorous final training pipeline.

## 9. Data Preparation and Preprocessing

### 9.1 Density map generation

Crowd counting in this project is based on density estimation rather than direct bounding-box detection. The core idea is:

- Each annotated head point contributes probability mass to a 2D density map.
- The sum of the density map approximates the total crowd count.

### 9.2 Fixed Gaussian approach

`scripts/preprocess_dataset.py` creates density maps from ShanghaiTech annotations using Gaussian smoothing over head points.

Basic steps:

1. Load image and ground-truth head points.
2. Create an empty map with the same image height and width.
3. Place impulses at head locations.
4. Apply Gaussian smoothing.
5. Save the result as `.npy`.

Observed generated assets:

- `data/density_maps`: 300 files
- `data/density_maps_test`: 182 files

### 9.3 Geometry-adaptive density maps

`scripts/dataset_v2.py` improves preprocessing by using a geometry-adaptive Gaussian strategy:

- For each head point, KDTree is used to estimate the average distance to neighboring heads.
- Sigma is computed from local neighbor spacing.
- Sigma is clamped between `1.5` and `10.0`.

The formula implemented is conceptually:

`sigma_i = clip(0.3 * avg_distance_to_3_neighbors, 1.5, 10.0)`

Why this matters:

- Dense scenes need smaller spread so nearby heads are not overly blurred.
- Sparse scenes need wider spread so isolated heads still contribute meaningful density.

### 9.4 Augmentation and training crops

The V2 dataset pipeline applies:

- multi-scale crop selection between `256` and `384`
- resize back to base size
- count-preserving density resizing
- horizontal flip
- brightness jitter

This improves generalization while preserving count consistency.

## 10. Model Architecture

The repository contains multiple counting paths.

### 10.1 Legacy baseline model

`training/train.py` implements a baseline crowd-density CNN:

- encoder: VGG16 feature extractor
- decoder: sequence of transposed convolutions
- output: 1-channel density map

This model predicts density only. Count is obtained by summing the density map and then calibrating it.

### 10.2 V2 dual-head model

`training/model_v2.py` implements the improved architecture:

- encoder: VGG16 features up to layer slice `[:24]`
- density decoder: transposed-convolution decoder
- regression head: global pooled feature to scalar count

The model outputs:

- `density`
- `direct_count`

The fused count is:

`final_count = 0.7 * density_sum + 0.3 * direct_count`

This design attempts to combine:

- local spatial evidence from the density branch
- global count awareness from the regression branch

### 10.3 LWCC DM-Count integration

The deployed app primarily uses `LWCC` if available. This is a strong practical choice because the saved project results show that LWCC performs far better than the current V2 branch.

The app uses scene-adaptive selection:

- `SHB` weights for sparse scenes
- `SHA` weights for denser scenes
- ensemble strategy in very dense scenes

The switch logic in `app.py` uses thresholds around:

- sparse threshold: `< 80`
- medium-to-dense zone: `80 to 200`
- very dense: `>= 200`

## 11. Training Strategy

### 11.1 Baseline training

The older baseline training path:

- uses ShanghaiTech A and B
- optionally mixes in Mall
- trains on CPU
- optimizes MSE over density maps
- validates mainly on ShanghaiTech Part B

This is useful as a baseline but not the strongest final system.

### 11.2 V2 training improvements

`training/train_v2.py` adds more advanced optimization ideas:

- focal-weighted MSE
- SSIM loss
- log-cosh count loss
- dynamic weighting by crowd-size bucket
- EMA validation model
- warmup scheduler
- cosine LR decay
- gradient accumulation
- early stopping

#### V2 loss design

The combined objective includes:

- density reconstruction term
- structure preservation term
- direct count supervision term

This is meant to reduce false negatives in sparse scenes and produce sharper density maps.

### 11.3 Stratified sampling

The V2 dataloader in `scripts/dataset_v2.py` builds a weighted sampler with target distribution:

- 40% sparse images with count `1-20`
- 30% medium images with count `21-50`
- 30% all remaining images

This is a practical design choice because sparse scenes are important for public-event monitoring and usually harder to balance during training.

## 12. Inference and Calibration

### 12.1 Baseline calibration

The baseline path uses calibration files:

- `models/calibration_small.json`
- `models/calibration_medium.json`
- `models/calibration_large.json`
- fallback `models/calibration.json`

The calibration formula is:

`predicted_count = exp(a) * raw_sum^b`

### 12.2 V2 inference

`training/inference_v2.py` performs:

- grayscale conversion
- tiled inference with tile size `256`
- 50% overlap
- optional horizontal-flip test-time augmentation
- density-map clipping and thresholding

### 12.3 V2 calibration

`training/calibrate_v2.py` uses two regimes:

- sparse calibration for `<= 100`
- dense calibration for `> 100`

Sparse calibration uses a simple scale relation:

`count = a * raw`

Dense calibration uses log-linear power-law fitting:

`count = exp(a) * raw^b`

## 13. Patch-Level Density Segmentation

This is the core idea that makes the project more useful than simple crowd counting.

Instead of analyzing the entire density map as one object, the system divides it into an `8 x 8` grid:

- total patches per scene: `64`

Each patch is converted into an 8-dimensional feature vector.

### 13.1 Features extracted per patch

The shared feature extractor in `app.py`, `training/segment_density.py`, and `training/evaluate.py` uses:

1. mean density
2. max density
3. standard deviation
4. coefficient of variation
5. max gradient magnitude
6. fraction of pixels above patch mean
7. normalized row index
8. normalized column index

These features capture:

- how dense a region is
- how concentrated or spread the density is
- how structured or abrupt the crowd layout is
- where the patch lies in the image

## 14. Unsupervised and Supervised Zone Analysis

### 14.1 KMeans clustering

KMeans is used as the original unsupervised segmentation baseline.

Process:

1. cluster all patch features into `k = 4`
2. sort cluster centers by mean density
3. assign labels:
   - `Low`
   - `Medium`
   - `High`
   - `Critical`

This gives an interpretable safety zoning scheme.

### 14.2 GMM soft clustering

Gaussian Mixture Model is used as a soft alternative to KMeans.

Benefits:

- patch membership probabilities
- confidence-aware labeling
- softer decision boundaries

### 14.3 DBSCAN anomaly detection

DBSCAN is used to identify unusual patches or density outliers.

This is useful when a few zones differ strongly from the rest of the scene and may deserve special attention.

### 14.4 XGBoost classifier

`training/train_classifier.py` trains an XGBoost classifier using KMeans-generated pseudo-labels.

Purpose:

- create a faster learned classifier for zone prediction
- reproduce KMeans-like decisions with a supervised model
- enable class probability estimation

## 15. Safety Analytics Layer

The safety layer converts model outputs into interpretable decisions.

### 15.1 Zone thresholds

The app supports 4 safety labels:

- `Low`
- `Medium`
- `High`
- `Critical`

The simple fallback mapping based on patch mean density is:

- `Low` if mean `< 0.1`
- `Medium` if mean `< 0.3`
- `High` if mean `< 0.6`
- `Critical` otherwise

### 15.2 Threat score

The app computes a threat score from zone statistics and crowd count. The implementation in `app.py` uses:

`threat = min(100, Critical*40 + High*20 + Medium*1 + min(15, crowd_count/10))`

Threat bands:

- `< 25`: `MINIMAL`
- `< 50`: `ELEVATED`
- `< 75`: `HIGH`
- `>= 75`: `CRITICAL`

### 15.3 Dynamic confidence

The app also computes a confidence score that blends:

- count-model confidence based on scene regime
- zone-classification confidence from the selected method

The implemented blend is:

`confidence = 0.7 * base_count_confidence + 0.3 * zone_confidence`

Base confidence depends on crowd size:

- sparse scenes get the highest base confidence
- denser scenes receive lower confidence because they are harder

### 15.4 Head-marker and portrait fallback

The dashboard contains extra logic for visualization robustness:

- local maxima are extracted from density maps for head-like markers
- portrait or close-up scenes are detected heuristically
- YuNet face detection is used as a fallback for close-up cases

This is useful because crowd-density models are not always ideal for small portrait-like scenes.

## 16. Streamlit Dashboard

The main application is implemented in `app.py`.

### 16.1 Main tabs

The dashboard includes the following tabs:

- `Live Analysis`
- `Compare Methods`
- `Dashboard`
- `Live Capture`
- `Batch Analysis`
- `RL Agent`

### 16.2 Core dashboard features

- upload and analyze single crowd images
- display crowd count
- show density heatmap overlay
- show safety-zone overlay
- compare KMeans, XGBoost, and GMM outputs
- show threat level and confidence
- analyze multiple images in batch
- allow phone-based live capture using QR-linked session access

### 16.3 Model loading

The app caches and loads:

- `models/best_model.pth`
- LWCC SHB and SHA weights
- `models/xgb_classifier.pkl`
- `models/kmeans_model.pkl`
- `models/gmm_model.pkl`
- `models/feature_scaler.pkl`

### 16.4 Actual live-analysis flow

The app follows this approximate order:

1. receive image
2. if LWCC is available, use LWCC crowd counting
3. otherwise, fall back to the baseline CNN and calibration
4. resize density map to image size
5. extract 64 patch feature vectors
6. classify patches using KMeans, XGBoost, or GMM
7. build safety overlay
8. compute threat and confidence
9. show metrics and visuals

## 17. Reinforcement Learning Module

The `RL Agent` tab in `app.py` is a demonstration-oriented decision-support module.

### 17.1 Network design

The DQN uses:

- input size: `69`
- output actions: `8`

State vector contents:

- 64 zone-risk values
- normalized crowd count
- normalized critical-zone count
- normalized high-zone count
- utilization-style feature
- normalized number of exits

### 17.2 Actions

The implemented action set includes:

- open all exits equally
- prioritize exits near critical zones
- close exits in critical zones
- one-way safe-side flow
- staged evacuation
- shelter in place
- emergency protocol
- gradual dispersal

### 17.3 Reward logic

Reward is based on:

- reduction in critical zones
- reduction in high zones
- penalty for evacuation time
- bonus for fully safe state
- additional penalty if the critical situation worsens

Important note:

This RL component is useful for demonstration and future-work discussion, but it should not be presented as a validated real-world evacuation controller.

## 18. Experimental Artifacts Present in the Repository

Saved model and result artifacts found locally include:

### 18.1 Models

- `models/best_model.pth`
- `models/best_model_v2.pth`
- `models/calibration.json`
- `models/calibration_small.json`
- `models/calibration_medium.json`
- `models/calibration_large.json`
- `models/calibration_v2.json`
- `models/xgb_classifier.pkl`
- `models/kmeans_model.pkl`
- `models/gmm_model.pkl`
- `models/dbscan_model.pkl`
- `models/feature_scaler.pkl`
- `models/face_detection_yunet_2023mar.onnx`

### 18.2 Results

- `results/eval_lwcc_detail.csv`
- `results/eval_lwcc_error_hist.png`
- `results/eval_lwcc_scatter.png`
- `results/eval_v2_detail.csv`
- `results/eval_v2_error_hist.png`
- `results/eval_v2_mae_buckets.png`
- `results/eval_v2_scatter.png`
- `results/prediction_result.png`
- safety map images under `results/safety_maps/`

## 19. Evaluation Methodology

The repository evaluates multiple layers of the system.

### 19.1 Counting metrics

Used metrics include:

- MAE
- RMSE
- bias
- estimated accuracy

The reported accuracy formula used in evaluation is:

`accuracy = max(0, 100 - (MAE / mean_GT) * 100)`

### 19.2 Patch-classification metrics

For zone models, the repository uses:

- accuracy
- weighted F1
- silhouette score
- cluster count
- anomaly ratio
- GMM confidence

## 20. Main Results From Current Repository State

### 20.1 LWCC DM-Count counting results

From `results/eval_lwcc_detail.csv`:

| Metric | Value |
|---|---:|
| Evaluated images | 498 |
| Overall MAE | 45.36 |
| Overall RMSE | 89.25 |
| Overall Bias | -26.90 |
| Overall Estimated Accuracy | 80.89% |

Sparse-scene range `1-100`:

| Metric | Value |
|---|---:|
| Images | 176 |
| MAE | 5.80 |
| RMSE | 9.26 |
| Bias | -4.26 |
| Accuracy | 90.25% |

Per-dataset MAE:

| Dataset | MAE |
|---|---:|
| ShanghaiTech Part B | 22.35 |
| ShanghaiTech Part A | 85.32 |

Interpretation:

- LWCC is clearly the strongest counting path in the current repository.
- It is especially effective in sparse and medium-density scenes.
- Dense scenes remain more challenging.

### 20.2 V2 custom model results

From `results/eval_v2_detail.csv`:

| Metric | Value |
|---|---:|
| Evaluated images | 798 |
| Overall MAE | 159.55 |
| Overall RMSE | 279.10 |
| Overall Bias | -159.36 |
| Overall Estimated Accuracy | 3.43% |

Sparse-scene range `1-100`:

| Metric | Value |
|---|---:|
| Images | 476 |
| MAE | 48.04 |
| RMSE | 54.02 |
| Bias | -47.73 |
| Accuracy | 5.28% |

Interpretation:

- The V2 branch is still undercounting heavily.
- It is better described as a research or improvement branch than as the final best model.

### 20.3 Zone-classification and segmentation results

From `training/evaluate.py` using saved models:

| Component | Metric | Value |
|---|---|---:|
| XGBoost | Accuracy | 0.9986 |
| XGBoost | Weighted F1 | 0.9986 |
| KMeans | Silhouette | 0.3119 |
| GMM | Silhouette | 0.2210 |
| GMM | Avg confidence | 0.9835 |
| DBSCAN | Clusters found | 8 |
| DBSCAN | Anomalies | 712 |
| DBSCAN | Total patches | 19200 |
| DBSCAN | Anomaly ratio | 0.0371 |

Interpretation:

- XGBoost reproduces KMeans-style zone labels very effectively.
- KMeans is a practical and interpretable unsupervised baseline.
- GMM adds soft confidence information.
- DBSCAN helps identify unusual local behavior.

## 21. Discussion and Analysis

This repository tells a clear technical story:

- The strongest practical counting backend is LWCC DM-Count.
- The strongest safety-zone reproduction model is XGBoost.
- The unsupervised clustering stage is valuable because it provides structure before any manual labeling.
- The dashboard is a major contribution because it turns model outputs into something usable for operators.

The project is therefore stronger as a **crowd-safety intelligence system** than as a pure custom crowd-counting model benchmark.

## 22. Strengths of the Project

- combines counting with safety interpretation
- supports multiple ML approaches in one system
- includes both unsupervised and supervised patch analysis
- offers real visual outputs rather than only metrics
- contains a deployable dashboard
- includes mobile capture and batch analysis
- uses practical scene-adaptive counting in the app

## 23. Limitations

- the custom V2 counting branch is not yet strong enough
- dense-scene performance is weaker than sparse-scene performance
- some legacy baseline dataset handling is simplified
- RL component is demonstration-oriented, not validated for real deployment
- `requirements.txt` is minimal and does not list every dependency used by the dashboard path

## 24. Practical Applications

- crowd monitoring in stadiums
- event safety monitoring in concerts and festivals
- surveillance support for police or security staff
- risk-zone visualization for control rooms
- crowd condition reporting during large public gatherings

## 25. Equal Work Division for 2 Members

This split keeps the workload balanced across modeling, analytics, and presentation.

### Member 1: Data, Density Maps, and Crowd Counting

Main responsibilities:

- understanding datasets and directory structure
- preparing density maps
- studying ShanghaiTech and JHU annotation formats
- explaining baseline model and V2 architecture
- explaining LWCC integration
- collecting count-based results and evaluation metrics
- preparing result graphs and count tables

Expected contribution outputs:

- dataset section
- preprocessing section
- counting-model section
- evaluation section for MAE, RMSE, and bias

### Member 2: Segmentation, Safety Logic, and Dashboard

Main responsibilities:

- explaining patch feature extraction
- implementing or presenting KMeans, GMM, DBSCAN, and XGBoost
- explaining threat score and confidence score
- covering the Streamlit dashboard and UI tabs
- preparing screenshots of overlays and safety maps
- explaining live capture, batch analysis, and RL demo

Expected contribution outputs:

- patch-segmentation section
- safety-analytics section
- dashboard section
- application and future-work section

### Why the split is equal

Member 1 covers:

- dataset engineering
- density learning
- crowd counting
- evaluation

Member 2 covers:

- patch analytics
- clustering and classification
- UI integration
- decision-support logic

Both are substantial and technically balanced.

## 26. Presentation Split for 2 Members

### Member 1: First half

- introduction
- problem statement
- motivation
- datasets
- preprocessing
- model architecture
- counting results

### Member 2: Second half

- segmentation methods
- safety logic
- dashboard features
- visual results
- limitations
- future work
- conclusion

## 27. How to Run the Project

Below is the practical execution flow based on repository scripts.

### 27.1 Install dependencies

At minimum the code uses packages such as:

- torch
- torchvision
- opencv-python
- numpy
- scipy
- pandas
- scikit-learn
- xgboost
- Pillow
- matplotlib
- seaborn
- joblib
- tabulate
- streamlit
- plotly
- lwcc

### 27.2 Generate density maps

```bash
python3 scripts/preprocess_dataset.py
```

### 27.3 Run segmentation models

```bash
python3 training/segment_density.py
```

### 27.4 Train patch classifier

```bash
python3 training/train_classifier.py
```

### 27.5 Run evaluation

```bash
python3 training/evaluate.py
python3 training/eval_lwcc.py
python3 training/eval_v2.py
```

### 27.6 Launch dashboard

```bash
python3 -m streamlit run app.py
```

## 28. Future Work

- improve dense-scene counting quality
- retrain and debug the V2 branch
- replace all placeholder legacy dataset logic with exact annotation parsing
- move from images to continuous video analysis
- add temporal smoothing and tracking
- improve calibration with larger validation sets
- make the RL module interact with a more realistic simulator

## 29. Conclusion

This project successfully demonstrates a complete crowd-analysis workflow for public-event safety. Its major contribution is not only crowd counting, but the conversion of density information into interpretable zone-level safety intelligence. The strongest current result comes from the LWCC DM-Count backend, while the segmentation and dashboard layers make the project useful for real demonstrations and practical public-safety discussion.

In short, the project shows how machine learning, unsupervised clustering, and interactive visualization can work together to support safer management of large crowds.

## 30. Short Final Summary for Submission

This project proposes a crowd-safety monitoring system that estimates crowd count, segments density into local safety zones, highlights critical areas, and presents the results in an interactive dashboard for public-event management.
