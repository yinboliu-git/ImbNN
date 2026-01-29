# Neural Collapse in Imbalanced Learning

A comprehensive research codebase for studying Neural Collapse (NC) phenomena in class-imbalanced learning scenarios. This project investigates how different models, loss functions, and training configurations affect the emergence of neural collapse patterns under class imbalance conditions.

## Features

- **16+ Model Architectures**: ResNet, VGG, ViT, MobileNet, EfficientNet, DenseNet, ConvNeXt, DINOv3, EVA-02, and more
- **Multiple Loss Functions**: Cross-Entropy, Focal Loss, Logit Adjustment, Balanced Softmax
- **Imbalanced Dataset Generation**: Configurable long-tailed distributions for MNIST, Fashion-MNIST, CIFAR-10/100, SVHN, ImageNet-LT
- **Neural Collapse Metrics**: Drift norm, overall accuracy, tail accuracy, layer-wise analysis
- **Distributed Training**: GPU job queue system for running large-scale experiments

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/anonymous/nc-imbalance.git
cd nc-imbalance

# Install the package
pip install -e .
```

### From Source

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- datasets >= 2.14.0 (for ImageNet-LT Parquet support)
- PyYAML >= 6.0
- NumPy >= 1.24.0

## Quick Start

### 1. Generate Experiment Configurations

```bash
# Generate all experiment configs
python scripts/gen_configs.py

# This creates configs in configs/exp1/, configs/exp2/, etc.
```

### 2. Run a Single Experiment

```bash
# Train with a specific config
python scripts/train.py --config configs/exp1/exp1_univ_cifar10_resnet18.yaml --save_root ./results
```

### 3. Run Full Experiment Suite

```bash
# Run all experiments with distributed GPU scheduling
bash scripts/run.sh
```

## Project Structure

```
nc-imbalance/
├── src/nc_imbalance/          # Main package
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   └── architectures.py   # ModelFactory, get_feature()
│   ├── data/                  # Dataset loaders
│   │   ├── __init__.py
│   │   └── imbalanced_dataset.py
│   ├── training/              # Loss functions
│   │   ├── __init__.py
│   │   └── losses.py
│   ├── analysis/              # NC metrics
│   │   ├── __init__.py
│   │   └── nc_metrics.py
│   └── utils/                 # Utilities
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── gen_configs.py        # Config generator
│   ├── run.sh                # Distributed runner
│   └── download_datasets.py  # Dataset checker
├── configs/                   # Generated configs
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── README.md
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Experiment Groups

The codebase includes 10 pre-configured experiment groups:

1. **exp1 - Universality**: Tests NC across 4 datasets × 14 models
2. **exp2 - Scaling Laws**: Validates drift-imbalance relationship across 10 imbalance factors
3. **exp3 - Loss Functions**: Compares CE, Focal, Logit Adjustment, Balanced Softmax
4. **exp4 - Weight Decay**: Ablation study on regularization strength
5. **exp5 - Optimizers**: Compares SGD, Adam, AdamW, RMSprop
6. **exp6 - Drift Penalty**: Tests explicit drift regularization
7. **exp7 - Batch Size**: Robustness to batch size variations
8. **exp8 - ViT Ablation**: LayerNorm impact on NC in Vision Transformers
9. **exp9 - Reproducibility**: Repeated runs for statistical significance
10. **exp10 - ImageNet-LT**: Large-scale experiments on ImageNet Long-Tail

## Configuration Format

```yaml
exp_name: "exp1_univ_cifar10_resnet18"
dataset:
  name: "cifar10"              # mnist, fmnist, cifar10, cifar100, imagenet_lt
  imb_factor: 0.01             # 1.0=balanced, 0.01=extreme, -1=native
model:
  name: "resnet18"             # See architectures.py for full list
train:
  lr: 0.1
  optimizer: "sgd"             # sgd, adam, rmsprop
  weight_decay: 1e-3
  loss: "ce"                   # ce, focal, logit_adjustment, balanced_softmax
  drift_penalty: 0.0           # Optional drift regularization
  epochs: 200
  batch_size: 256
  analyze_freq: 2              # Compute metrics every N epochs
  analyze_layers: false        # Enable layer-wise drift tracking
  save_logits: false           # Save raw logits for analysis
```

## Supported Models

### Classic CNNs
- ResNet-18/50 (with optional LayerNorm/BatchNorm variants)
- VGG-11/16
- MobileNet-V2
- ShuffleNet-V2
- RegNet-Y-400MF
- DenseNet-121
- EfficientNet-B0
- GoogLeNet
- ConvNeXt-Tiny

### Vision Transformers
- ViT-Tiny (modified for NC)
- ViT-Tiny-Original (control group)

### Modern Architectures
- MobileNetV4-Small
- RepViT-M1
- DINOv3-Small
- EVA-02-Tiny

## Output Structure

```
results/
└── exp1_univ_cifar10_resnet18/
    ├── metrics_epoch_2.json       # Accuracy, tail accuracy, drift norm
    ├── metrics_epoch_200.json
    ├── raw_geo_epoch_2.pt         # Predictions, targets, drift vectors
    ├── raw_geo_epoch_200.pt
    └── final_model.pth            # Trained model weights
```

## Advanced Usage

### Custom Experiments

```python
from nc_imbalance import ModelFactory, ImbalancedDatasetGenerator, NCAnalyzer

# Create imbalanced dataset
data_gen = ImbalancedDatasetGenerator(name='cifar10', imb_factor=0.01)
loader = data_gen.get_loader(batch_size=256)

# Create model
model = ModelFactory.get_model('resnet18', num_classes=10)

# Analyze neural collapse
analyzer = NCAnalyzer(model, loader, device, save_dir='./results', model_name='resnet18')
analyzer.compute_all_metrics(epoch=200)
```

### Adding New Models

1. Add model creation logic to `ModelFactory.get_model()` in `src/nc_imbalance/models/architectures.py`
2. Add feature extraction logic to `get_feature()` in the same file
3. Ensure the final classifier uses `bias=False`

### Custom Loss Functions

```python
from nc_imbalance.training import LogitAdjustmentLoss

# Use in training
criterion = LogitAdjustmentLoss(cls_num_list=data_gen.img_num_list, tau=1.0)
```

## Key Implementation Details

### Model Architecture Constraints
- All classifiers use `bias=False` to facilitate NC analysis
- ViT models require `encoder.ln = nn.Identity()` to observe NC
- Modern models (EVA-02, DINOv3) use CLS token pooling

### Feature Extraction
- `get_feature()` returns features AFTER normalization layers but BEFORE the final classifier
- Each architecture requires custom handling (ViT uses CLS token, CNNs use GAP output)

### Imbalanced Datasets
- `imb_factor` controls severity: 1.0 (balanced) to 0.01 (extreme imbalance)
- Exponential decay creates long-tailed distributions
- `img_num_list` attribute provides per-class sample counts

## Citation

If you use this code in your research, please cite:

```bibtex
@article{anonymous2024nc,
  title={Neural Collapse in Imbalanced Learning},
  author={Anonymous},
  journal={Under Review},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

This is an anonymous submission for peer review. Contributions will be accepted after the review process.

## Acknowledgments

This research builds upon the neural collapse literature and uses models from PyTorch, timm, and other open-source libraries.
