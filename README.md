Here is the updated README code. It focuses solely on the high-level overview and the specific "Why" behind the project, as requested.

```markdown
# 🛡️ S.A.F.E. (Structural Anonymization for Facial Exclusion)

> **Surgical-Grade Brain Extraction & Medical Image Anonymization Pipeline**
---

## 📖 Overview

**S.A.F.E.** is a high-performance, GPU-accelerated pipeline designed for **automated brain extraction (skull stripping)** and **facial anonymization** in medical imaging (MRI).

It moves beyond standard "black-box" Deep Learning approaches by utilizing a **Dual-Specialist Architecture** powered by XGBoost and GPU-accelerated Computer Vision (CuPy/OpenCV). The system treats brain segmentation as a surgical procedure: identifying core tissue with a **Classifier** and refining edges with a **Regressor**, effectively separating the brain from non-brain tissues like the skull and face.

### 🌟 Why S.A.F.E.? (The Need for Structural Anonymization)

As the name **Structural Anonymization for Facial Exclusion** suggests, this tool addresses a critical privacy concern in medical data sharing: **Re-identification Risk.**

* **Facial Reconstruction Risk:** Modern rendering techniques can reconstruct a patient's face from a standard MRI scan (T1-weighted images), potentially violating patient privacy and HIPAA/GDPR regulations.
* **Beyond Pixel Blurring:** Simple blurring or blacking out of the face often destroys crucial structural data needed for analysis. S.A.F.E. performs **Structural Exclusion**, surgically removing the facial structure and skull while preserving the complete geometry of the brain tissue.
* **Data Sharing & Research:** To share medical datasets publicly or between institutions, robust de-identification is mandatory. S.A.F.E. ensures that the shared data contains **only the brain**, rendering facial reconstruction impossible while keeping the scientific value of the data intact.

---

## ⚡ Installation

### Prerequisites
* **OS:** Linux / Windows (WSL2 recommended)
* **GPU:** NVIDIA GPU with CUDA support (Required)
* **Python:** 3.8+

### Dependencies
Install the required libraries. Note that `cupy` must match your CUDA version (e.g., `cupy-cuda11x` or `cupy-cuda12x`).

```bash
# Core requirements
pip install numpy pandas scikit-learn scikit-image opencv-python-headless
pip install nibabel pydicom simpleitk ants dicom2nifti tqdm matplotlib

# GPU Acceleration (Check your CUDA version!)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install cupy-cuda11x  # Or cupy-cuda12x
pip install cucim         # NVIDIA RAPIDS image processing
pip install xgboost       # GPU-enabled XGBoost

```

---

## 💻 Usage

### Quick Start

To run S.A.F.E. on your dataset:

1. Open the script `safe_pipeline.py`.
2. Scroll to the `__main__` block at the bottom.
3. Adjust the loop or `INPUT_TARGET` path to point to your NIfTI/DICOM data.
4. Run:

```bash
python safe_pipeline.py

```

---

## 📂 Project Structure

```text
S.A.F.E/
├── safe_pipeline.py     # Main executable script
├── mni_icbm152_...      # (Optional) MNI Template files for registration
├── NFBS_Dataset/        # (Optional) Dataset folder
└── README.md            # You are here

```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
<b>Developed by Akhil</b>




<i> IISc Bangalore | M.Tech (Research) - CDS</i>
</p>

```

```
