# Transmissions Project

A research-oriented Python project for simulating, analyzing, and predicting transmission profiles in graphene-based quantum devices.

## Folder Structure

- `src/physics/` – Green's functions, hopping matrices, tight-binding Hamiltonians  
- `src/models/` – Machine learning models (PyTorch)  
- `src/train/` – Training loops, physics-informed losses  
- `src/data/` – Dataset utilities + preprocessing  
- `models/` – Saved neural network checkpoints (ignored by Git)  
- `data/` – Large datasets (ignored by Git)  
- `notebooks/` – Jupyter analysis notebooks  
- `scripts/` – CLI scripts for running jobs  

## Getting Started

Install requirements:

```bash
pip install -r requirements.txt
```

Run training:

```bash
bash scripts/train.sh
```
