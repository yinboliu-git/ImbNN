import yaml
import os
import numpy as np

def save_config(config, group_name, name):
    path = f'configs_re/{group_name}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{name}.yaml', 'w') as f:
        yaml.dump(config, f)

# Base configuration
COMMON_TRAIN = {
    "lr": 0.1,
    "optimizer": "sgd",
    "weight_decay": 1e-3,
    "loss": "ce",
    "epochs": 200,
    "batch_size": 256,
    "analyze_freq": 1,
    "analyze_layers": False
}

# =========================================================
# Experiment A (Exp-SOTA): SOTA method comparison
# =========================================================
# Target: ResNet18, CIFAR100-LT (rho=100), 4 loss functions
methods = [
    {"loss": "ce", "name": "baseline_ce"},
    {"loss": "focal", "name": "focal"},
    {"loss": "logit_adjustment", "name": "logit_adj"},
    {"loss": "balanced_softmax", "name": "balanced_softmax"}
]

# for m in methods:
#     cfg = {
#         "exp_name": f"re_sota_{m['name']}",
#         "dataset": {"name": "cifar100", "imb_factor": 0.01},
#         "model": {"name": "resnet18"},
#         "train": COMMON_TRAIN.copy()
#     }
#     cfg['train']['loss'] = m['loss']

#     # Enable layer-wise analysis (Ana-Layer) and logit saving (Ana-Gamma)
#     # Only enable detailed analysis in SOTA experiments for App K.3 and K.4
#     cfg['train']['analyze_layers'] = True
#     cfg['train']['save_logits'] = True
    
#     save_config(cfg, "sota", cfg["exp_name"])

# # =========================================================
# # Experiment B (Exp-Corr): Correlation analysis
# # =========================================================
# # Target: Random grid search rho [10, 200] -> imb [0.1, 0.005], lambda [1e-4, 1e-2]
# np.random.seed(42)
# num_samples = 20

# for i in range(num_samples):
#     # Random sampling
#     rho = np.random.uniform(10, 200)
#     imb_factor = 1.0 / rho
#     # Log-uniform sampling for weight decay
#     wd = 10 ** np.random.uniform(-4, -2) 
    
#     cfg = {
#         "exp_name": f"re_corr_sample_{i}",
#         "dataset": {"name": "cifar100", "imb_factor": float(imb_factor)},
#         "model": {"name": "resnet18"},
#         "train": COMMON_TRAIN.copy()
#     }
#     cfg['train']['weight_decay'] = float(wd)
#     cfg['train']['analyze_freq'] = 200
    
#     save_config(cfg, "corr", cfg["exp_name"])

# =========================================================
# Experiment C (Exp-Causal): Drift elimination intervention
# =========================================================
# Target: Introduce Drift Penalty, eta=0 vs eta=0.5

etas = [0.3, 0.5]
models = ['resnet18']
for m in models:
    for eta in etas:
        cfg = {
            "exp_name": f"re_causal_eta_{m}_{eta}",
            "dataset": {"name": "cifar100", "imb_factor": 0.01},
            "model": {"name": m},
            "train": COMMON_TRAIN.copy()
        }
        if eta > 0:
            cfg['train']['drift_penalty'] = eta

        save_config(cfg, "causal", cfg["exp_name"])

vit_variants = ["vit_tiny", "vit_tiny_orig"]
for m in vit_variants:
    cfg = {
        "exp_name": f"re_causal_eta_{m}",
        "dataset": {"name": "cifar100", "imb_factor": 0.01},
        "model": {"name": m},
        "train": COMMON_TRAIN.copy()
    }
    save_config(cfg, "causal", cfg["exp_name"])

print("Generated Reviewer Experiments Configurations in 'configs_re/'")