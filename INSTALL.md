# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM for ImageNet-LT experiments

## Installation Methods

### Method 1: Pip Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/anonymous/nc-imbalance.git
cd nc-imbalance

# Install in editable mode
pip install -e .
```

### Method 2: Manual Installation

```bash
# Install dependencies
pip install torch>=2.0.0 torchvision>=0.15.0
pip install timm>=0.9.0 datasets>=2.14.0
pip install PyYAML>=6.0 numpy>=1.24.0 Pillow>=9.0.0

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n nc-imbalance python=3.10
conda activate nc-imbalance

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install timm datasets PyYAML numpy Pillow

# Install package
pip install -e .
```

## Verify Installation

```bash
# Test imports
python -c "from nc_imbalance import ModelFactory, ImbalancedDatasetGenerator, NCAnalyzer; print('Installation successful!')"

# Check datasets
python scripts/download_datasets.py

# Generate test configs
python scripts/gen_configs.py
```

## GPU Setup

The code automatically detects and uses CUDA if available. For multi-GPU setups, the `run.sh` script distributes jobs across 4 GPUs by default.

To modify GPU allocation:
```bash
# Edit scripts/run.sh
NUM_GPUS=4           # Number of GPUs
TASKS_PER_GPU=3      # Concurrent tasks per GPU
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure the package is installed:
```bash
pip install -e .
```

### CUDA Out of Memory
Reduce batch size in config files:
```yaml
train:
  batch_size: 128  # Reduce from 256
```

### Dataset Download Issues
Manually do datasets to `./datasets/` directory following the structure in `download_datasets.py`.

## Development Setup

For development with testing and linting tools:
```bash
pip install -e ".[dev]"
```

This installs additional tools:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)
