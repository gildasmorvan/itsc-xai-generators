# ITSC-XAI-Generators

Explainable AI for inter-turn short-circuit (ITSC) fault detection in low-speed synchronous generators.

---

## 📊 Dataset

The dataset used in this repository is publicly available on Zenodo:

https://doi.org/10.5281/zenodo.19230572

It contains frequency-domain features extracted from stray magnetic field measurements collected on six industrial generators.

---

## ⚙️ Setup

To run the notebooks, follow these steps:

### 1. Create a virtual environment
    python3 -m venv venv

### 2. Activate the environment

- Linux / Mac:
    source venv/bin/activate

- Windows:
    venv\Scripts\activate

### 3. Install dependencies
    pip install -r requirements.txt

### 4. Launch JupyterLab
    jupyter-lab

---

## 📓 Notebooks

This repository includes the following notebooks:

- [ClassicAlgorithms.ipynb ](ClassicAlgorithms.ipynb)
  Evaluates classical machine learning algorithms on the dataset using cross-machine splits.

- [Raw.ipynb](Raw.ipynb)
  Evaluates the XAI-based approach using raw frequency-domain features.

- [Sign_Of_First_Order_Difference.ipynb](Sign_Of_First_Order_Difference.ipynb)
  Evaluates the XAI-based approach using the sign of the first-order difference of the features.
  
- [First_Order_Difference.ipynb](First_Order_Difference.ipynb)
  Evaluates the XAI-based approach using the first-order difference of the features.

---

## 🎯 Purpose

This project investigates the robustness of explainable models for fault detection under domain shift across machines, and highlights the limitations of classical data-driven approaches in this setting.

---

## 📄 License

This project is released under the MIT License (see LICENSE file).

---

## 🙏 Acknowledgements

 This work has benefited from the support of EDF and the National Research Agency under France 2030, MAIA Project (ANR-22-EXES-0009).
 
If you use this code or dataset, please consider citing the associated work.


```bibtex
@dataset{edf_dataset,
  author    = {Morvan, Gildas and Ducreux, Jean-Pierre and Jacq, Thierry and Pusca, Remus and Romary, Raphael},
  title     = {Frequency-Domain Dataset for Inter-Turn Short-Circuit (ITSC) Fault Detection in EDF Low-Speed Synchronous Generators},
  month     = mar,
  year      = 2026,
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.19230572}
}