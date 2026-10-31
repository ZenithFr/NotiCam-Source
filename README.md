# NotiCam Source Workspace

A comprehensive real-time portrait bokeh system designed for the MediaTek Dimensity 7350 Pro (`GalagaIND` / `A001`, Mali-G615 MC2, APU 655).

---

## 🗂️ Directory Organization

| Directory | Purpose | Contents |
|---|---|---|
| **[`noticam-app/`](noticam-app)** | **Mobile Application (Git Submodule)** | Standalone Android app with native Vulkan 1.3 bokeh engine, C++ JNI, shaders, Shizuku Camera2 service, build scripts (`build.sh`), and signed APK in `release/`. Submodule tracked from [NotiCam](https://github.com/ZenithFr/NotiCam). |
| **[`auxcam-app/`](noticam-app)** | **Compatibility Symlink** | Symlink to `noticam-app` preserving backwards compatibility with existing path references. |
| **[`apks/`](apks)** | **Compiled Binaries** | Packaged and aligned production debug APK (`NotiCam-debug.apk`). |
| **[`kaggle/`](kaggle)** | **Kaggle GPU Training & Dataset Pipeline** | Automated multi-stage matting network training notebook (`auxcam_portrait_net_training.ipynb`), `build_kaggle_notebook.py`, modular `ml_training/` scripts, and deduplicated 15,435 corpus metadata. |
| **[`scripts/`](scripts)** | **Hardware Diagnostics & Probing Scripts** | Reverse-engineering and calibration tools: `depth_probe/`, `depth_semantics/`, NNAPI op delegation probes, motion noise tests, and CoC validation suites. |
| **[`captures/`](captures)** | **Raw Hardware Dumps & Crop Audits** | 4K raw Y-plane buffers, full-resolution RGBA captures, synced Y8 disparity dumps, and Laplacian variance verification crops. |
| **[`docs/`](docs)** | **Engineering Logs & Benchmark Reports** | Comprehensive forensic audit and physical device benchmarks ([`RESULTS.md`](docs/RESULTS.md)). |
| **[`packages/`](packages)** | **Prebuilt Libraries & Dependencies** | TensorFlow Lite 2.16.1 AAR archives and extracted headers/JNI binaries. |

---

## 📦 Submodule Initialization

When cloning this repository, initialize the `noticam-app` submodule with:
```bash
git clone --recurse-submodules https://github.com/ZenithFr/NotiCam-Source.git
# Or if already cloned:
git submodule update --init --recursive
```
