# AI-Powered Flood Detection System

An end-to-end computer vision system designed to automate disaster response by analyzing satellite imagery. This system generates real-time, deterministic flood risk reports across diverse infrastructure topologies by combining land cover classification with pixel-perfect surface water segmentation.

## 🧠 System Architecture

This project utilizes a dual-model deep learning approach, shifting away from single-source limitations to ensure robust, industrial-level inference:

1. **Land Cover Classification:** Analyzes the structural environment (e.g., Highway, Industrial, Permanent Crop) to determine inherent infrastructure vulnerability.
2. **Surface Water Semantic Segmentation:** Detects and isolates floodwater boundaries at a pixel level to calculate the exact percentage of water coverage.
3. **Deterministic Risk Engine:** Fuses the outputs of the two neural networks. By indexing the vulnerability of the classified terrain against the segmented water ratio, the engine automatically thresholds and flags **CRITICAL** risk zones.

## 📂 Repository Structure

* `model.py`: Contains the core deep learning model architectures (Classification and Segmentation).
* `train.py`: Contains the custom training loops, data loaders, and validation logic.

## 📊 Datasets

**Note:** Due to GitHub's file size limits, the raw dataset files are not included in this repository. To train or test the models, you will need to acquire the following industry-standard datasets and extract them into a local `processed/` directory:

* **EuroSAT:** Used for high-confidence land cover and terrain classification.
* **Sen1Floods11:** Used for training the semantic segmentation models to detect surface water.
* **DeepGlobe:** Integrated for topographical context and structural extraction.
* **Elevation Data:** Used to supplement structural analysis.

*Expected local directory structure before running `train.py`:*
```text
Flood_Risk_System/
│
├── model.py
├── train.py
└── processed/
    ├── deepglobe/
    ├── elevation/
    ├── EuroSAT/
    └── sen1floods11/
