# AffNet: Link Prediction and Recommendation

This repository contains the implementation of **AffNet** models for:

- **Link Prediction** (`affNet`)
- **Recommendation** (`AffNetR`)

The project uses **PyTorch**, **PyTorch Geometric**, and **TensorFlow**, and has been tested on **Windows with Python 3.10**.  
Due to GPU- and platform-specific dependencies, some packages (notably PyTorch and PyTorch Geometric CUDA extensions) must be installed **manually**.

---

## Repository Structure

```
AffNet_LP/
├── affNet/        # Link Prediction module
│   ├── predict_link.py
│   └── utils.py
├── AffNetR/       # Recommendation module
│   ├── predict_rec.py
│   └── utils.py
├── requirements.txt
├── run_all.sh
└── README.md
```

- **affNet/**: Code for link prediction experiments  
- **AffNetR/**: Code for recommender system experiments  

---

## Requirements

- **OS**: Windows (tested)
- **Python**: 3.10.x
- **GPU**: Optional (CUDA 11.8 recommended for PyTorch)
- **pip**: Latest version

⚠️ TensorFlow GPU support on Windows is only available up to **TensorFlow 2.10.1**.

---

## Installation

### Step 1: Create and activate a virtual environment

From the project root:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

### Step 2: Copy project files

Copy the project files into the new folder where the virtual environment was created (if not already there).

---

### Step 3: Install PyTorch (CUDA 11.8)

PyTorch is installed **manually** to ensure correct CUDA support.

```powershell
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation:

```powershell
python -m pip install torch==2.1.0
```

---

### Step 4: Install core dependencies

```powershell
python -m pip install -r requirements.txt
```

> Note: `requirements.txt` intentionally excludes CUDA-specific PyTorch and PyG wheels.

---

### Step 5: Install additional PyTorch ecosystem packages

#### TorchVision and TorchAudio (CUDA 11.8)

```powershell
python -m pip install torchvision==0.16.0 torchaudio==2.1.0 `
  --index-url https://download.pytorch.org/whl/cu118
```

#### PyTorch Geometric dependencies (CUDA 11.8)

```powershell
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv `
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

python -m pip install torch-geometric
```

---

### Step 6: Configure dataset paths

For **link prediction**, edit the following variables in:

```
affNet/predict_link.py
```

For **recommendation**, edit the corresponding script in:

```
AffNetR/
```

Set the following variables near the top of the script:

```python
root = "<path to module directory>"
data_folder = "<path to dataset directory>"
```

---

### Step 7: Run experiments

#### Link Prediction

```powershell
python affNet/predict_link.py --dataset=Roman-empire --emb_features=128 --n_heads=4 --max_nodes=4000 --init_lr=0.0005 --epochs=2000
```

#### Recommendation

```powershell
python AffNetR/predict_rec.py
```

Alternatively, run all experiments:

```powershell
bash run_all.sh
```

(On Windows, this may require WSL or Git Bash.)

---

## NumPy and TensorFlow Compatibility

- TensorFlow **2.10.1** is **not compatible with NumPy 2.x**
- This project pins NumPy to `< 2`

If you encounter NumPy-related errors:

```powershell
python -m pip install "numpy<2"
```

---

## Tested Environment

- Python: 3.10.x  
- PyTorch: 2.1.0 (CUDA 11.8)  
- TensorFlow: 2.10.1  
- NumPy: 1.26.x  
- matplotlib: 3.8.4  

---

## License

Add license information here if applicable.
