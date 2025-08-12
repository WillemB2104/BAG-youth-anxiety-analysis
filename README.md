---
# Brain Age Examinations in Youth Anxiety Disorders  
**Author:** Willem Bruin

---

## Overview

This repository contains two main components for brain age estimation and analysis based on FreeSurfer (FS)-derived MRI data:

1. **Python Jupyter Notebook**  
   Implements the **MCCQRNN** model to estimate brain age with uncertainty adjustment using FreeSurfer features.

2. **R Analysis Script**  
   Processes model outputs to compute Brain Age Gap (BAG) metrics, perform group-level statistical analyses, and generate visualizations.

---

## Python Notebook: MCCQRNN Brain Age Modeling

This notebook performs the following steps:

- Prepares FreeSurfer-derived MRI features from the ENIGMA-ANXIETY working group  
- Trains the MCCQRNN model on healthy controls using cross-validation  
- Applies the trained model to unseen patient data  
- Computes uncertainty-adjusted Brain Age Gap (BAG) scores  
- Generates plots to evaluate model performance  
- Conducts occlusion sensitivity mapping to identify brain regions contributing to predictions  

**Model reference:**  
Hahn et al. (2022). *An uncertainty-aware, shareable, and transparent neural network architecture for brain-age modeling.* Science Advances, 8(1), eabg9471.  
[https://www.science.org/doi/10.1126/sciadv.abg9471](https://www.science.org/doi/10.1126/sciadv.abg9471)

**Source code for MCCQRNN:**  
[https://github.com/wwu-mmll/uncertainty-brain-age](https://github.com/wwu-mmll/uncertainty-brain-age)

The MCCQRNN model version used here is included in `/Scripts/MCCQRNN_Regressor.py`.

---

## R Script: Statistical Analyses

This script uses the Python-generated outputs to:

- Load and preprocess input data (FreeSurfer morphometric measures, MCCQRNN predictions, clinical and demographic variables)  
- Perform linear mixed-effects models for group comparisons  
- Conduct transdiagnostic and subgroup analyses  
- Test clinical associations within patient groups  
- Perform occlusion sensitivity mapping to highlight region-specific contributions to BAG estimations  

---

## Workflow

1. Run the **Python notebook** to train and apply the MCCQRNN model, generating brain age estimates with uncertainty adjustment and occlusion sensitivity data.  
2. Run the **R script** to perform statistical analyses, visualize results, and interpret region-specific effects from occlusion sensitivity mapping.

---

## Requirements

- Python environment with required packages (see: [requirements.txt](https://github.com/wwu-mmll/mccqrnn_docker/blob/main/docker/requirements.txt))  
- R environment with packages for mixed-effects modeling and plotting

---

## Contact

For questions or collaboration, please contact Willem Bruin.

---
