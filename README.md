# Heart Failure Detection

---

## Installation

Set up a Python 3.12 virtual environment and install dependencies:


```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

---

## ️ Project Structure

Code is located under the `src/` folder with the following main components:

- `model/`: Network definitions (e.g., Inception (ours), Attia, Cho)
- `dataset/`: Dataset loading and preprocessing - you need to modify this if you want to train your own model
- `config/`: YAML config files for model training
- `scripts/`: Helper scripts (e.g., plotting Ray results)
- `train.py`: Main training script

---

##  How to train a model

Activate your Python environment and run training with a specific config file:

```bash
python src/train.py --config src/config/inception.yml
```

---

##  Using the Inception Ensemble

To use an ensemble of Inception models (e.g., trained on 5 folds), import and instantiate `InceptionEnsemble` as follows:

```python
import torch
from src.model.inception_ensemble import InceptionEnsemble

# Define path to weights (e.g., trained on 5 folds)
weights_dir = "weights"

ensemble = InceptionEnsemble(weights_dir=weights_dir)

# Create dummy input: (batch_size=7, channels=8, signal_length=5000) # 500 Hz times 10 sec
x = torch.randn(7, 8, 5000)  # NOTE: model input should be z-normalized
y = ensemble(x)  # Shape: (7, 5, 1) since 5 models and 1 output class

```
---
