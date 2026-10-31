# Diagnostic & Calibration Scripts

A collection of hardware telemetry probes, reverse-engineering utilities, and test scripts used to calibrate the camera and compute subsystems on the CMF Phone 2 Pro.

---

## 🛠️ Modules

- **`depth_probe/`**: Privileged standalone APK project and shell scripts to probe MediaTek physical Camera ID 5 disparity/depth stream metadata.
- **`depth_semantics/`**: Calibration utilities for baseline, focal length, disparity-to-depth ($Z$) conversion curves, and letterbox geometry mapping.
- **`tests/`**:
  - `test_ema_motion_noise.py`: Verification of exponential moving average filtering on depth streams across motion dynamics.
  - `test_pre_kaggle_gate.py`: Pre-flight verification gates (TFRecord round-trip, ISP augmentations, INT8 APU latency).
  - `verify_stub_alpha_pipeline.py`: Stub alpha pipeline validation.
- **`nnapi_op_probe.cpp`**: MediaTek APU / NNAPI accelerator op delegation probe.
- **`coc_gpu_test`**: Standalone Vulkan compute executable to test Circle of Confusion execution on Mali-G615.
