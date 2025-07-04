# **SEED: Speaker Embedding Enhancement Diffusion Model**
## Accepted at Interspeech 2025 | Pytorch Implementation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license) [![Python Version](https://img.shields.io/badge/python-3.8%2B-green)](#requirements) [![Coverage Status](https://img.shields.io/badge/coverage-99%25-brightgreen)](#)

---

## 🚀 Overview

**SEED** is a first diffusion-based embedding enhancement framework designed to improve speaker representation robustness under adverse acoustic conditions. It leverages a powerful diffusion network paired with state-of-the-art speaker representation models (ResNetSE34V2, ECAPA-TDNN) to refine original speaker embeddings into more noise-robust speaker representations.

We believe that the SEED framework can be applied to various representation models (e.g., for Speech Recognition, Speech Emotion Recognition, Face Recognition, etc.), not just for speaker recognition tasks.

<p align="center">
  <img src="docs/main_figure.png" alt="SEED framework" />
</p>

### ✨ Key Features

* **Lightweight and Simple**: Easily applied to all speaker representation models, including ResNetSE34V2, ECAPA-TDNN, and WavLM-ECAPA, etc.
* **No Speaker Labels Required**: Can be trained on any clean speech data without explicit labels.

---

## 📦 Contents

1. [Requirements](#-requirements)
2. [Datasets](#-datasets)
3. [Training](#️-training)
4. [Evaluation](#-evaluation)
5. [Configuration Reference](#️-configuration-reference)
6. [Extending SEED](#️-extending-seed-to-your-own-representation-model)
7. [Pretrained Models](#-pretrained-models)
8. [Utilities & Troubleshooting](#-utilities--troubleshooting)
9. [Citation](#-citation)

---

## 📋 Requirements

* **OS**: Linux
* **Python**: 3.8+
* **System Tools**: `wget`, `ffmpeg`
* **CUDA Toolkits**: 12.5.0 
* **Pytorch**: 2.1.2
> **Note**: In author's environment, we use `conda install nvidia/label/cuda-12.5.0::cuda-toolkit` (https://anaconda.org/nvidia/cuda-toolkit)

Install Python dependencies:

```bash
pip install -r requirements.txt
sudo apt-get install wget ffmpeg
```

---

## 📊 Datasets

### Prepare SEED Training and Evaluation Datasets

First, please read [datasets/README.md](datasets/README.md) for more details.
You can make all datasets by following the instruction in [datasets/README.md](datasets/README.md).

### Training Dataset Summary

SEED is trained on the following clean speech datasets and audio-augmentation datasets:

* **LibriTTS-R**  (`train-clean-100` + `train-clean-360`, \~460h)
* **Libri-Light** (`small`, \~577h)
> **Note**: SEED does *not* require speaker labels. Provide a manifest file listing `<file_path>` per line.

* **MUSAN** (Music, Speech, and Noise)
* **RIRs**  (Room Impulse Responses)
> **Note**: SEED use these datasets for audio-augmentation (simulating noisy speech data from clean speech data).

```text
# Example: For libritts + librilight (1,000h), we make a manifest file like this:
train_libritts+librilight_1000h.txt
/path/to/libritts-R_16k/1241/103_1241_000071_000000.wav
/path/to/libritts-R_16k/1241/1040_133433_000157_000000.wav
```

### Evaluation Dataset Summary

* **VoxCeleb1 (For validation of training results)**
* *VC-Mix* & *VoxSRC23* for environmental robustness benchmarks 


Manifests are located under `datasets/manifests/`:
```bash
datasets/
├── manifests/
│   ├── train_libritts+light_1000h.txt
│   ├── vox1-O.txt
│   ├── vcmix_test.txt
```

---


## 🏋️ Training

### ResNetSE34V2 (freezed) -> speaker embedding -> SEED (trainable)

```bash
python main.py \
  --config configs/ResNetSE34V2_SEED_rdmmlp3.yaml \
  --save_path exps/resnetse34v2_SEED_rdmmlp3
```

* **Backbone**: `ResNetSE34V2`
* **Diffusion**: `rdm_mlp`, layers=3
* **Timesteps**: Train=1000, Sample=50
* **Loss**: L1
* **Self-Conditioning**: Enabled

### ECAPA-TDNN (freezed) -> speaker embedding -> SEED (trainable)

```bash
python main.py \
  --config configs/ECAPA_TDNN_SEED.yaml \
  --save_path exps/ecapa_tdnn_SEED \
  --wandb     # Enable wandb logging (optional)
```

#### Tips

* `--mixedprec` for mixed-precision for fp16 training (To fast training).
* `--distributed` for DDP (set like `CUDA_VISIBLE_DEVICES=0,1,2,3`). 
* `--wandb` for experiment logging with Weights & Biases. It is optional. Configure with `--project`, `--entity`, `--group`, and `--name` parameters.
  ```bash
  --wandb                        # Enable wandb logging (optional)
  --project "SEED"               # Wandb project name
  --entity "your_wandb_entity"   # Your wandb username or team name
  --group "experiments"          # Group related experiments together
  --name "ecapa_seed_baseline"   # Specific experiment name
  ```
> **Note**: In this paper, we didn't use `--mixedprec` and `--distributed` options.

#### 🔧 Troubleshooting
**Key Mismatch Errors with Your Own Backbone?**
>
If you're using your own pretrained model and encounter errors about mismatched keys, it's likely due to a `state_dict` prefix issue (e.g., `module.` or `your_previous_classname.`) from a different training setup.
>
Don't worry, we have an easy fix! Head over to our [**Utilities & Troubleshooting**](#-utilities--troubleshooting) guide to resolve this in just a few steps.

---

## 🧪 Evaluation

### Diffusion & Backbone Combined

```bash
# ResNetSE34V2
python main.py \
  --eval \
  --config configs/ResNetSE34V2_SEED_rdmmlp3.yaml \
  --pretrained_backbone_model pretrained/official_resnetse34V2.model \
  --pretrained_diffusion_model pretrained/resnet34V2_SEED_evalseed_2690.model \
  --seed 2690
  --train_diffusion True
```

```bash
# ECAPA-TDNN
python main.py \
  --eval \
  --config configs/ECAPA_SEED_rdmmlp3.yaml \
  --pretrained_backbone_model pretrained/official_ecapatdnn.model \
  --pretrained_diffusion_model pretrained/ecapa_SEED_evalseed_898.model \
  --seed 898
  --train_diffusion True
```

### Backbone Only (Baseline)

```bash
python main.py \
  --eval \
  --config configs/ResNetSE34V2_baseline.yaml \
  --pretrained_backbone_model pretrained/official_resnetse34V2.model \
  --test_path datasets/voxceleb1 \
  --test_list datasets/manifests/vox1-O.txt \
  --train_diffusion False
```

---

## ⚙️ Configuration Reference

```yaml
# Speaker Backbone
model: ResNetSE34V2
batch_size: 300
nOut: 512
pretrained_speaker_model: ./pretrained/official_resnetse34V2.model

# Diffusion
train_diffusion: True
diffusion_network: rdm_mlp
diffusion_num_layers: 3
train_timesteps: 1000
sample_timesteps: 50
self_cond: True

# Optimizer
lr: 5e-4
optimizer: adamW
```

---

## 🛠️ Extending SEED to your own representation model

1. **Add new backbones** in `models/yourtask/your_backbone.py`.
2. Customizing `DatasetLoader.py` for your own dataset.
3. Prepare your own training datasets (with augmentation strategies) and evaluation datasets.
4. Follow existing modules as templates.
5. Run main.py!

---


## 📂 Pretrained Models

> Official checkpoints can be downloaded under `pretrained/`.

For SEED, we provide the following pretrained models:
* **ResNetSE34V2** (backbone): `official_resnetse34V2.model`
* **-->**          (SEED-Diffusion): `resnet34V2_SEED_evalseed_2690.model`
* **ECAPA-TDNN**   (backbone): `official_ecapa_tdnn.model`
* **-->**          (SEED-Diffusion): `ecapa_SEED_evalseed_898.model`

---

## 🧰 Utilities & Troubleshooting

### Easily Handle Your Own Backbone Weights

In SEED, we officially load backbone weights into `self.backbone`. However, when using your own pretrained models, you often encounter loading conflicts due to mismatched keys in the `state_dict`.

**Common Problem:**
- Your model has keys like `module.layer1.conv.weight` (from DDP training)
- Or keys like `your_previous_classname.conv2d.layer1.weight` (from different class structure)
- But SEED expects keys like `layer1.conv.weight` to load into `self.backbone`

**Our Solution:**
We provide a simple utility code `model_params_tool.py` to easily modify the key values of neural network model weight files. This tool automatically detects and removes problematic prefixes, allowing you to seamlessly integrate your backbone weights with SEED.

#### 1. Analyzing Model Prefixes (`analyze_prefix`)

First, analyze your model's `state_dict` structure:

```bash
python model_params_tool.py analyze_prefix --model_path path/to/your/model.model
```

**Example Output (Clean case):**
```text
--- Prefix Analysis Results ---
Model Path: pretrained/official_resnetse34V2.model
Total Keys: 245
Has 'module.' Prefix: False
Most Common Prefix Found: N/A
Expected Prefix for Check: Not specified
Prefix Consistency: True
Sample Keys (first 10):
  - layer1.0.conv1.weight
  - layer1.0.bn1.weight
  - layer1.0.bn1.bias
...
--- End of Analysis ---
```

**Example Output (When a prefix mismatch occurs):**
```text
--- Prefix Analysis Results ---
Model Path: path/to/your/problematic_model.ckpt
Total Keys: 245
Has 'module.' Prefix: False
Most Common Prefix Found: your_previous_classname.
Expected Prefix for Check: Not specified
Prefix Consistency: True
Sample Keys (first 10):
  - your_previous_classname.layer1.0.conv1.weight
  - your_previous_classname.layer1.0.bn1.weight
  - your_previous_classname.layer1.0.bn1.bias
...
--- End of Analysis ---
```

#### 2. Removing Model Prefixes (`remove_prefix`)

**Remove a specific prefix:**
```bash
python model_params_tool.py remove_prefix \
  --model_path path/to/your/problematic_model.ckpt \
  --output_path path/to/your/refined_model.ckpt \
  --target_prefix "your_previous_classname."
```

- **Overwrite the original file:**
If you don't specify an `--output_path`, the changes will overwrite the original model file.
```bash
python model_params_tool.py remove_prefix \
  --model_path path/to/your/original_model.ckpt \
  --target_prefix "unwanted_prefix."
```

Using this tool, you can easily modify model checkpoints trained in various environments to fit your current project.

---

## 📜 Citation

If you use SEED in your research, please cite:

```bibtex
@article{nam2025seed,
  title={SEED: Speaker Embedding Enhancement Diffusion Model},
  author={Nam, KiHyun and Heo, Jungwoo and Jung, Jee-weon and Park, Gangin and Jung, Chaeyoung and Yu, Ha-Jin and Chung, Joon Son},
  journal={arXiv preprint arXiv:2505.16798},
  year={2025}
}
```

---
