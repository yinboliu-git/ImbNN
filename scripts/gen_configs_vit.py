import yaml
import os

def save_config(config, group_name, name):
    path = f'configs_vit/{group_name}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{name}.yaml', 'w') as f:
        yaml.dump(config, f)

# =========================================================
# ViT specialized experiment configuration (5 repetitions)
# =========================================================

COMMON_TRAIN = {
    "lr": 0.05,
    "optimizer": "sgd",
    "weight_decay": 5e-4,
    "loss": "ce",
    "epochs": 100,
    "batch_size": 128,
    "analyze_freq": 5
}

experiments = [
    # Experimental group: BatchNorm (follows Scaling Law)
    {"model": "vit_tiny",      "exp_id": "vit_bn"},
    # Control group: LayerNorm (original version)
    {"model": "vit_tiny_orig", "exp_id": "vit_ln"}
]

NUM_RUNS = 5

for run_idx in range(NUM_RUNS):
    # Set different base seed for each run to ensure different but reproducible randomness
    current_seed = 1000 + run_idx

    for exp in experiments:
        cfg = {
            # Add run suffix to name
            "exp_name": f"check_{exp['exp_id']}_run{run_idx}",
            "seed": current_seed,
            "dataset": {"name": "cifar100", "imb_factor": 0.01},
            "model": {"name": exp['model']},
            "train": COMMON_TRAIN.copy()
        }

        save_config(cfg, "vit_check", cfg["exp_name"])

print(f"Generated {len(experiments) * NUM_RUNS} ViT Configs (5 runs each) in 'configs_vit/'")