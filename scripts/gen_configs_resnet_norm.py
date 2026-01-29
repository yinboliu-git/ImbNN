import yaml
import os

def save_config(config, group_name, name):
    path = f'configs_resnet_norm/{group_name}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{name}.yaml', 'w') as f:
        yaml.dump(config, f)

# Base configuration
COMMON_TRAIN = {
    "lr": 0.1,
    "optimizer": "sgd",
    "weight_decay": 1e-3,
    "loss": "ce",
    "epochs": 100,
    "batch_size": 256,
    "analyze_freq": 10
}

# Experiment design: Compare three ResNet18 variants
# 1. resnet18 (Standard): No Norm before FC
# 2. resnet18_ln: LayerNorm before FC (Testing Straitjacket)
# 3. resnet18_bn: BatchNorm1d before FC (Testing Inverse Scaling)

variants = [
    "resnet18",
    "resnet18_ln",
    "resnet18_bn"
]

NUM_RUNS = 3

for run_idx in range(NUM_RUNS):
    seed = 2025 + run_idx
    for m in variants:
        cfg = {
            "exp_name": f"check_{m}_run{run_idx}",
            "seed": seed,
            "dataset": {"name": "cifar100", "imb_factor": 0.01},
            "model": {"name": m},
            "train": COMMON_TRAIN.copy()
        }

        save_config(cfg, "resnet_norm", cfg["exp_name"])

print(f"Generated {len(variants)*NUM_RUNS} ResNet Norm configs in 'configs_resnet_norm/'")