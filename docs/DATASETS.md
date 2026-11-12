# Datasets

All five datasets are freely accessible (no institutional credentialing required). Verify
the license text on each page before any use beyond research/education, since Kaggle
mirror pages occasionally restate licenses loosely.

## X-Ray — Chest X-Ray Images (Pneumonia)

- Source: Kaggle, `paultimothymooney/chest-xray-pneumonia` (Kermany, Zhang, Mooney mirror)
- Original paper: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by
  Image-Based Deep Learning," *Cell*, 2018.
- License: CC BY 4.0 (verify on the Kaggle dataset page before reuse)
- Access: Kaggle API, `kaggle datasets download paultimothymooney/chest-xray-pneumonia`

## CT — NoduleMNIST3D (MedMNIST v2)

- Source: `pip install medmnist`, auto-downloads from Zenodo on first use
- Derived from: LIDC-IDRI (The Cancer Imaging Archive) via the LUNA16 nodule-candidate
  preprocessing pipeline
- License: CC BY 4.0 (MedMNIST v2 license; LIDC-IDRI itself is CC BY 3.0)
- Citation: Yang et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification," *Scientific Data*, 2023.

## MRI — Brain Tumor MRI Dataset

- Source: Kaggle, `masoudnickparvar/brain-tumor-mri-dataset`
- License: listed as CC0 (Public Domain) on Kaggle; verify before reuse
- Access: Kaggle API, `kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset`
- Classes: glioma, meningioma, pituitary, no tumor

## Ultrasound — BUSI (Breast Ultrasound Images Dataset)

- Source: Kaggle, `aryashah2k/breast-ultrasound-images-dataset`
- Original paper: Al-Dhabyani et al., "Dataset of breast ultrasound images," *Data in
  Brief*, 2020.
- License: open for research use; verify exact terms before reuse
- Classes: normal, benign, malignant

## ECG — ECG Heartbeat Categorization Dataset

- Source: Kaggle, `shayanfazeli/heartbeat`
- Derived from: MIT-BIH Arrhythmia Database and PTB Diagnostic ECG Database (PhysioNet)
- Original paper: Kachuee et al., "ECG Heartbeat Classification: A Deep Transferable
  Representation," IEEE ICHI, 2018.
- License: PhysioNet source databases are under the Open Data Commons Attribution License
  (ODC-BY); verify the Kaggle mirror's restated terms before reuse
- Classes: 5-class heartbeat categorization (normal + 4 arrhythmia types)
