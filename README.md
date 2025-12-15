
# Heart Failure Detection

---

## ️ Project Structure

Code is located under the `src/` folder with the following main components:

- `model/`: Network definitions (e.g., Inception (ours), Attia, Cho)
- `dataset/`: Dataset loading and preprocessing, you need to modify this if you want to train your own model
- `config/`: YAML config files for model training **and** data preparation/inference
- `scripts/`: Helper scripts (e.g., plotting Ray results)
- `train.py`: Main training script

---

##  How to train a model

Activate your Python environment and run training with a specific config file:

```bash
python -m src.train --config src/config/inception.yml
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
x = torch.randn(7, 8, 5000)  # NOTE: model input should be z-normalized per lead
# NOTE: lead order is aVL, aVF, V1, V2, V3, V4, V5, V6
y = ensemble(x)  # Shape: (7, 5, 1) since 5 models and 1 output class
```

---
# Evaluating the model on MIMIC-IV

## Preparing Data for Inference (Step 1)

Download the necessary datasets, i.e. MIMIC-IV, MIMIC-IV-ECG and MIMIC-IV-ECG-Ext-ICD from https://physionet.org/.

Use the YAML config at `src/config/prepare-mimiciv-ntprobnp.yaml` (edit paths as needed). Then run:

```bash
python -m src.scripts.connect_bloodtest_and_ecg --config src/config/prepare-mimiciv-ntprobnp.yaml
```

This writes a single CSV (specified in the config file) that links ECGs with NTproBNP.

> [!NOTE]  
> This step can take several minutes due to the large blood sample file.

---

## Running Inference (Step 2)

Use the YAML config at `src/config/mimiciv-inference.yaml` and run:

```bash
python -m src.scripts.run_inference_mimiciv --config src/config/mimiciv-inference.yaml
```
The expected output is

```text
Target 0 AUC: 0.866 (95% CI: 0.861-0.871, n=161352) 
Target 1 AUC: 0.901 (95% CI: 0.895-0.908, n=140720)
Target 2 AUC: 0.957 (95% CI: 0.950-0.964, n=4879)
```

and individual predictions will be saved as CSV in the folder specified in the config file.

> [!NOTE]  
> This step can also take several minutes and is much faster using a GPU. Inference takes less than 40 seconds on a NVIDIA RTX 5090 GPU.
