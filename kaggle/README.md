# AuxCam Portrait Net — Kaggle Training Suite

Automated multi-stage deep learning pipeline for training the lightweight portrait matting network (`AuxCamPortraitNet`, 189,090 parameters) on Nvidia T4 GPU instances.

---

## 📂 Contents

- **`auxcam_portrait_net_training.ipynb`**: Master executable Kaggle notebook.
- **`build_kaggle_notebook.py`**: Deterministic generator for the master notebook.
- **`kernel-metadata.json`**: Kaggle CLI kernel configuration.
- **`ml_training/`**: Modular Python modules:
  - `model_mnv2_lrlf.py`: Architectural definition with ASPP, low-rank linear fusion, and edge preservation.
  - `losses.py`: Weighted composite loss ($\mathcal{L}_{\alpha} + \mathcal{L}_{grad} + \mathcal{L}_{lap}$).
  - `augment.py`: Real-time ISP simulation (Gaussian noise, WB shifts, exposure jitter) and spatial transforms.
  - `train_stage1.py`: Stage 1 training (frozen backbone stages 0–4, 137,282 trainable parameters).
  - `train_stage2.py`: Stage 2 full unfreeze with cosine decay ($10^{-4} \to 10^{-6}$).
  - `export_int8.py`: Representative calibration and INT8 PTQ quantization for MediaTek APU deployment.
- **`eval_set/`**: Real CMF Phone 2 Pro test stills.
- **`smoke_augmented_samples/`**: Verification samples for data augmentation invariants.

---

## 📊 Dataset Ingestion & Deduplication Summary

| Dataset | Ingested Pairs | Unique (SHA-1) | Train | Val | Provenance / Role |
|---|---|---|---|---|---|
| **P3M-10k** | 10,421 | 10,421 | 9,379 | 1,042 | Primary portrait matting ground truth |
| **AIM-500** | 500 | 409 | 368 | 41 | Salient object & portrait test benchmark |
| **HHM** | 2,000 | 2,000 | 1,800 | 200 | HIM-2K human-instance matting substitution |
| **PMD** | 2,605 | 2,605 | 2,344 | 261 | Figaro1k + RMAS online composition matting proxy |
| **Total** | **15,526** | **15,435** | **13,891** | **1,544** | Deterministic hash-based split (zero twin leakage) |
