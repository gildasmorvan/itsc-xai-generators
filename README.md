# itsc-xai-generators
Explainable AI for inter-turn short-circuit fault detection in low-speed synchronous generators

## Dataset

The dataset used in this repo is available on (Zenodo)[https://doi.org/10.5281/zenodo.19230572].

## How to run the notebooks

1. Create a Python virtual environment
```
python3 -m venv venv
```
2. Activate it

	* Linux/Mac: `source venv/bin/activate`
	* Windows: `venv\Scripts\activate`
	
3. Install the dependencies

```
pip install -r requirements.txt
```

4. Launch Jupyter-lab

```
jupyter-lab
```

## Notebooks

The repo contains two notebooks.

* (ClassicAlgorithms.ipynb)[ClassicAlgorithms.ipynb] evaluates the performances of classical classification algorithms on the dataset.

* (XAI.ipynb)[XAI.ipynb] evaluates the performances of our XAI-based approach.