# COMPLETE PROJECT REPORT

## Unsupervised Crowd Density Segmentation for Public Event Safety

---

**Project Type:** Major Project / Capstone  
**Domain:** Computer Vision · Machine Learning · Public Safety  
**Technology Stack:** Python, PyTorch, OpenCV, Streamlit, Plotly, Scikit-learn, XGBoost  
**Application:** SafeCrowd Vision — Real-Time Crowd Safety Intelligence Dashboard  
**Codebase Size:** ~6,500 lines (app.py) + 18 training/evaluation scripts  
**Repository:** Unsupervised-Crowd-Density-Segmentation-for-Public-Event-Safety  

---

## Table of Contents

1. Abstract
2. Introduction
3. Problem Statement
4. Motivation
5. Objectives
6. Scope of the Project
7. Literature Survey
8. System Architecture
9. Technology Stack & Tools
10. Datasets Used
11. Data Preprocessing & Density Map Generation
12. Model Architecture — Crowd Counting
13. Training Strategy
14. Inference & Calibration
15. Patch-Level Density Segmentation
16. Unsupervised & Supervised Zone Classification
17. Safety Analytics Layer
18. Reinforcement Learning Module
19. Streamlit Dashboard — SafeCrowd Vision
20. Dashboard Features Deep Dive
21. Mobile Live Capture System
22. Repository Structure
23. Experimental Results
24. Evaluation Methodology
25. Discussion & Analysis
26. Strengths
27. Limitations
28. Practical Applications
29. Future Work
30. How to Run the Project
31. Equal Work Division
32. Presentation Plan
33. Conclusion
34. References

---

## 1. Abstract

This project develops **SafeCrowd Vision**, an intelligent crowd-analysis system for public-event safety. The system accepts a crowd image, estimates the number of people using deep learning (DM-Count via LWCC), generates a spatial density map, divides the scene into 64 local zones using an 8×8 grid, and classifies those zones into four safety levels: **Low**, **Medium**, **High**, and **Critical**.

The classification employs both unsupervised methods (KMeans, GMM, DBSCAN) and a supervised XGBoost classifier trained on pseudo-labels. A threat score, confidence metric, evacuation time estimate, and venue capacity monitor are computed in real-time. A Deep Q-Network (DQN) reinforcement learning agent recommends optimal evacuation strategies.

All outputs are presented in a premium dark-themed Streamlit dashboard with 6 interactive tabs, mobile camera integration via ngrok, batch analysis, and a fullscreen presentation mode. The system achieves **~81% overall counting accuracy** (MAE 5.80 for sparse scenes) and **99.86% zone classification accuracy** (XGBoost F1).

---

## 2. Introduction

Crowd management is one of the most critical safety challenges at large public gatherings — concerts, rallies, festivals, sports events, religious gatherings, and political programs. Crowd disasters such as stampedes often originate not from overall overcrowding, but from **localized dense pockets** that go undetected until it is too late.

Traditional crowd monitoring relies on human observers watching CCTV feeds. This approach suffers from fatigue, subjectivity, delayed reaction, and inability to quantify risk levels. A computer-vision-based system can assist security teams by continuously estimating crowd density and automatically highlighting high-risk zones.

This project goes beyond simple crowd counting. It builds a complete **crowd-safety intelligence pipeline** that converts raw images into actionable safety decisions — counting people, mapping density spatially, classifying risk zones, scoring threats, estimating evacuation time, and recommending actions through a reinforcement learning agent.

---

## 3. Problem Statement

**How can we automatically analyze crowd images and convert them into useful safety intelligence for public-event monitoring?**

The system must answer three practical questions:

1. **How many people are present?** — Crowd count estimation
2. **Which local areas are becoming overcrowded?** — Spatial density segmentation
3. **How risky is the current scene?** — Threat scoring and safety classification

---

## 4. Motivation

- Public events require **rapid detection** of overcrowded regions before stampede situations develop
- Security teams need **visual, explainable outputs** — not just numbers
- Manual counting is **unreliable** in scenes with 50+ people
- Early warnings can **reduce the chance of stampede-like situations** by minutes
- A **usable dashboard** is necessary for real-world deployment and demonstration
- Existing crowd-counting research focuses on accuracy metrics but rarely provides **actionable safety intelligence**

---

## 5. Objectives

1. Build a crowd counting system using density estimation (DM-Count)
2. Generate geometry-adaptive density maps from head annotations
3. Divide scenes into 64 local zones and analyze patch-level density
4. Compare unsupervised (KMeans, GMM, DBSCAN) and supervised (XGBoost) methods for zone classification
5. Compute threat scores, confidence levels, and evacuation time estimates
6. Build a reinforcement learning agent for evacuation policy recommendation
7. Create a premium safety-aware dashboard (SafeCrowd Vision) for analysis and visualization
8. Support mobile camera integration, batch processing, and presentation mode
9. Demonstrate how crowd intelligence can support public-event safety decisions

---

## 6. Scope of the Project

**In scope:**
- Image-based crowd analysis (single images)
- Density-map generation from head annotations
- Deep learning crowd counting (DM-Count / LWCC)
- Unsupervised clustering (KMeans, GMM, DBSCAN)
- Supervised classification (XGBoost)
- Safety visualization with interactive dashboard
- Mobile camera integration via ngrok tunneling
- Reinforcement learning evacuation demo
- Batch image analysis
- JSON report export

**Out of scope:**
- Real-time video stream processing
- Production-grade surveillance deployment
- Validated evacuation control system
- Multi-camera tracking

---

## 7. Literature Survey

### 7.1 Crowd Counting Approaches

| Approach | Description | Limitations |
|----------|-------------|-------------|
| Detection-based | Detect individual heads/bodies using object detectors | Fails in dense crowds due to occlusion |
| Regression-based | Map image features directly to count | Loses spatial information |
| **Density estimation** | Generate spatial density map; sum = count | Best balance of accuracy and spatial info |

### 7.2 Key Models Referenced

| Model | Paper | Contribution |
|-------|-------|-------------|
| MCNN | Zhang et al., 2016 | Multi-column CNN for multi-scale features |
| CSRNet | Li et al., 2018 | Dilated convolutions for dense prediction |
| **DM-Count** | Wang et al., 2020 | Distribution matching for crowd counting |
| CAN | Liu et al., 2019 | Context-aware network |

### 7.3 Why DM-Count (LWCC)

DM-Count uses **optimal transport** (distribution matching) loss instead of pixel-wise MSE, which handles annotation noise better and produces sharper density maps. The LWCC library provides pre-trained DM-Count weights for ShanghaiTech A and B datasets, making it practical for deployment.

### 7.4 Zone Classification Literature

- **KMeans** — Lloyd's algorithm for hard clustering (MacQueen, 1967)
- **GMM** — Expectation-Maximization for soft probabilistic clustering (Dempster et al., 1977)
- **DBSCAN** — Density-based clustering for anomaly detection (Ester et al., 1996)
- **XGBoost** — Gradient-boosted trees for supervised classification (Chen & Guestrin, 2016)

---

## 8. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Crowd Image                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 1: CROWD COUNTING                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ DM-Count SHB │  │ DM-Count SHA │  │  Ensemble    │      │
│  │ (sparse <80) │  │ (80-200)     │  │  (dense>200) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  Output: crowd_count + density_map                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 2: PATCH-LEVEL SEGMENTATION                 │
│  • 8×8 grid → 64 patches                                   │
│  • 8 features per patch (mean, max, std, CoV, gradient,     │
│    above-mean fraction, row index, col index)               │
│  Output: 64 feature vectors                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 3: ZONE CLASSIFICATION                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ KMeans  │  │  GMM    │  │ DBSCAN  │  │ XGBoost │      │
│  │(k=4)    │  │(k=4)   │  │         │  │         │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│  Output: Low / Medium / High / Critical per zone            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 4: SAFETY ANALYTICS                         │
│  • Threat Score (0-100%)                                    │
│  • Dynamic Confidence (%)                                   │
│  • Venue Capacity Utilization                               │
│  • Evacuation Time Estimate                                 │
│  • Alert Classification (Minimal/Elevated/High/Critical)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 5: RL EVACUATION AGENT                      │
│  • DQN (69-dim state → 8 actions)                           │
│  • Epsilon-greedy exploration                               │
│  • Experience replay with target network                    │
│  Output: Recommended evacuation strategy                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 6: DASHBOARD (SafeCrowd Vision)             │
│  6 Tabs: Live Analysis │ Compare │ Dashboard │              │
│          Live Capture │ Batch │ RL Agent                    │
│  Features: Overlays, Gauges, Charts, QR, JSON Export        │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Technology Stack & Tools

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.9+ | Core implementation |
| **Deep Learning** | PyTorch, torchvision | DM-Count, VGG16, DQN |
| **Computer Vision** | OpenCV (cv2) | Image processing, YuNet face detection |
| **Crowd Counting** | LWCC library | Pre-trained DM-Count weights |
| **ML Clustering** | scikit-learn | KMeans, GMM, DBSCAN, StandardScaler |
| **ML Classification** | XGBoost | Supervised zone classifier |
| **Dashboard** | Streamlit | Web application framework |
| **Visualization** | Plotly | Interactive charts, gauges |
| **Data Processing** | NumPy, Pandas, SciPy | Arrays, dataframes, Gaussian filters |
| **Image Processing** | Pillow (PIL) | Image format handling |
| **Serialization** | joblib, JSON | Model and config persistence |
| **Tunneling** | ngrok | Mobile camera access |
| **Face Detection** | YuNet (ONNX) | Portrait/close-up fallback |

---

## 10. Datasets Used

### 10.1 ShanghaiTech Part A (SHA)
- **Training images:** 300
- **Test images:** 182
- **Characteristic:** Dense crowd scenes (avg ~500 people/image)
- **Use:** Dense-scene training and evaluation

### 10.2 ShanghaiTech Part B (SHB)
- **Training images:** 400
- **Test images:** 316
- **Characteristic:** Sparse to medium crowd scenes (avg ~120 people/image)
- **Use:** Primary evaluation dataset, app presentation metrics

### 10.3 JHU Crowd V2.0
- **Training images:** 2,272 (filtered to 1,076 with count ≤ 100)
- **Validation images:** 500 (filtered to 258)
- **Test images:** 1,600 (filtered to 737)
- **Use:** V2 pipeline sparse-scene training

### 10.4 UCSD and Mall
- Referenced in older baseline training scripts
- Used as supplementary small-crowd datasets

---

## 11. Data Preprocessing & Density Map Generation

### 11.1 Core Concept
Instead of detecting individual people, density estimation places a Gaussian kernel at each annotated head location. The resulting density map has the property that **integrating (summing) over any region gives the estimated count** for that region.

### 11.2 Fixed Gaussian Approach
**Script:** `scripts/preprocess_dataset.py`

1. Load image and ground-truth head point annotations
2. Create empty density map matching image dimensions
3. Place unit impulses at each head location
4. Apply Gaussian smoothing (σ = 15)
5. Save as `.npy` file

**Generated assets:**
- `data/density_maps/` — 300 training density maps
- `data/density_maps_test/` — 182 test density maps

### 11.3 Geometry-Adaptive Gaussian (V2)
**Script:** `scripts/dataset_v2.py`

Uses KDTree to compute adaptive sigma per head point:

```
σ_i = clip(0.3 × avg_distance_to_3_nearest_neighbors, 1.5, 10.0)
```

**Why this matters:**
- Dense scenes → smaller σ → prevents over-blurring of nearby heads
- Sparse scenes → larger σ → isolated heads still contribute meaningful density

### 11.4 Data Augmentation (V2)
- Multi-scale crop selection (256–384 px)
- Count-preserving density resizing
- Horizontal flip
- Brightness jitter (±10%)
- Stratified sampling: 40% sparse (1–20), 30% medium (21–50), 30% other

---

## 12. Model Architecture — Crowd Counting

### 12.1 Legacy Baseline Model
**Script:** `training/train.py`

| Component | Architecture |
|-----------|-------------|
| Encoder | VGG16 feature extractor (pretrained ImageNet) |
| Decoder | Transposed convolution stack |
| Output | 1-channel density map |
| Count | Sum of density map + power-law calibration |

### 12.2 V2 Dual-Head Model
**Script:** `training/model_v2.py`

| Component | Architecture |
|-----------|-------------|
| Encoder | VGG16 features (layers [:24]) |
| Density Head | Transposed-convolution decoder → density map |
| Count Head | Global average pooling → FC → scalar count |
| **Fused Count** | **0.7 × density_sum + 0.3 × direct_count** |

This dual-head design combines:
- **Local spatial evidence** from the density branch
- **Global count awareness** from the regression branch

### 12.3 LWCC DM-Count (Primary — Deployed)
**The app primarily uses LWCC DM-Count** with scene-adaptive weight selection:

| Scene Type | Crowd Count | Weights Used |
|-----------|-------------|-------------|
| Sparse | < 80 | SHB (ShanghaiTech Part B) |
| Medium-Dense | 80–200 | SHA (ShanghaiTech Part A) |
| Very Dense | > 200 | Ensemble (SHA + SHB average) |

DM-Count uses **optimal transport loss** (distribution matching) instead of pixel-wise MSE, producing better density maps with less annotation noise sensitivity.

---

## 13. Training Strategy

### 13.1 V2 Training Pipeline
**Script:** `training/train_v2.py`

| Feature | Implementation |
|---------|---------------|
| **Loss Function** | Focal-weighted MSE + SSIM + Log-cosh count loss |
| **Optimizer** | AdamW with weight decay |
| **LR Schedule** | Warmup (5 epochs) + Cosine decay |
| **Regularization** | Gradient accumulation (4 steps), EMA validation model |
| **Early Stopping** | Patience-based on validation MAE |
| **Batch Strategy** | Stratified sampling (40/30/30 split by density) |

### 13.2 Loss Components

| Loss | Weight | Purpose |
|------|--------|---------|
| Focal MSE | Primary | Density map reconstruction with focus on dense regions |
| SSIM | 0.1 | Structural similarity preservation |
| Log-cosh Count | 0.5 | Direct count regression (smooth L1 alternative) |

---

## 14. Inference & Calibration

### 14.1 Baseline Calibration
Power-law correction: `predicted_count = exp(a) × raw_sum^b`

Calibration files: `calibration_small.json`, `calibration_medium.json`, `calibration_large.json`

### 14.2 V2 Tiled Inference
**Script:** `training/inference_v2.py`
- Grayscale conversion
- Tiled inference (256×256 tiles, 50% overlap)
- Horizontal-flip test-time augmentation (TTA)
- Density-map clipping and thresholding

### 14.3 LWCC Inference (Primary)
- Direct call to `lwcc.count(image, model_name="DM-Count", model_weights="SHB")`
- Returns both count and full-resolution density map
- Scene-adaptive weight switching in `app.py`

---

## 15. Patch-Level Density Segmentation

This is the **core innovation** that makes the project more useful than simple crowd counting.

### 15.1 Grid Division
The density map is divided into an **8×8 grid**, producing **64 patches** per scene.

### 15.2 Feature Extraction (8 features per patch)

| # | Feature | Description | What It Captures |
|---|---------|-------------|-----------------|
| 1 | Mean density | Average density value | Overall crowd level |
| 2 | Max density | Peak density value | Hotspot intensity |
| 3 | Standard deviation | Density spread | Uniformity of crowd |
| 4 | Coefficient of variation | std/mean ratio | Relative variability |
| 5 | Max gradient magnitude | Steepest density change | Edge of crowd boundaries |
| 6 | Above-mean fraction | % pixels above patch mean | Density distribution shape |
| 7 | Normalized row index | i / 7.0 | Spatial position (vertical) |
| 8 | Normalized col index | j / 7.0 | Spatial position (horizontal) |

Total feature matrix per image: **64 × 8 = 512 values**

---
