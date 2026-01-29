import yaml
import os

def save_config(config, group_name, name):
    path = f'configs/{group_name}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{name}.yaml', 'w') as f:
        yaml.dump(config, f)

# =========================================================
# Base definitions & global best practices configuration
# =========================================================
datasets = ["mnist", "fmnist", "cifar10", "cifar100"]
models = [
    "vgg11", "vgg16", "resnet18", "resnet50",
    "mobilenet_v2", "shufflenet_v2", "regnet_y_400mf",
    "vit_tiny", "densenet121", "efficientnet_b0", "googlenet",
    "mobilenetv4_small", "repvit_m1", "dinov3_small"
]

model_list = ["vgg11", "resnet18", "vit_tiny"]
data_list = ["cifar10", "cifar100"]
# Common training configuration
COMMON_TRAIN_CONFIG = {
    "lr": 0.1,
    "optimizer": "sgd",
    "weight_decay": 1e-3,
    "loss": "ce",
    "epochs": 200,
    "batch_size": 256,
    "analyze_freq": 2
}

# =========================================================
# Experiment 1: Universality validation
# =========================================================
# Count: 5 * 16 = 80
for d in datasets:
    for m in models:
        cfg = {
            "exp_name": f"exp1_univ_{d}_{m}",
            "dataset": {"name": d, "imb_factor": 0.01},
            "model": {"name": m},
            "train": COMMON_TRAIN_CONFIG.copy()
        }
        save_config(cfg, "exp1", cfg["exp_name"])

# =========================================================
# Experiment 9: Universality validation + repeated experiments
# =========================================================
# Count: 5 * 16 = 80
for r in range(1, 6):
    for d in data_list:
        for m in model_list:
            cfg = {
                "exp_name": f"exp9_univ_{d}_{m}_repeat_{r}",
                "dataset": {"name": d, "imb_factor": 0.01},
                "model": {"name": m},
                "train": COMMON_TRAIN_CONFIG.copy()
            }
            save_config(cfg, "exp9", cfg["exp_name"])

# =========================================================
# Experiment 2: Scaling law validation
# =========================================================
imb_factors = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
# Count: 10 * 3 = 30
for d in data_list:
    for f in imb_factors:
        for m in model_list:
            cfg = {
                "exp_name": f"exp2_scaling_{m}_{d}_f_{f}",
                "dataset": {"name": d, "imb_factor": f},
                "model": {"name": m},
                "train": COMMON_TRAIN_CONFIG.copy()
            }
            save_config(cfg, "exp2", cfg["exp_name"])

# =========================================================
# Experiment 3: Loss functions
# =========================================================
# Count: 4 * 3 = 12
losses = ["ce", "mse", "label_smoothing", "focal"]
for d in data_list:
    for l in losses:
        for m in model_list:
            cfg = {
                "exp_name": f"exp3_loss_{m}_{d}_{l}",
                "dataset": {"name": d, "imb_factor": 0.01},
                "model": {"name": m},
                "train": COMMON_TRAIN_CONFIG.copy()
            }
            cfg['train']['loss'] = l
            if l == "mse":
                cfg['train']['lr'] = 0.2
            save_config(cfg, "exp3", cfg["exp_name"])

# =========================================================
# Experiment 4: Regularization mechanism (Weight Decay)
# =========================================================
# Count: 6
wd_list = [0, 1e-4, 5e-4, 1e-3, 5e-3]
for d in data_list:
    for wd in wd_list:
        for m in model_list:
            cfg = {
                "exp_name": f"exp4_wd_{m}_{d}_{wd}",
                "dataset": {"name": d, "imb_factor": 0.01},
            "model": {"name": m},
            "train": COMMON_TRAIN_CONFIG.copy()
            }
            cfg['train']['weight_decay'] = wd
            save_config(cfg, "exp4", cfg["exp_name"])

# =========================================================
# Experiment 5: Optimizer influence
# =========================================================
# Count: 4 * 3 = 12
optimizers = ["sgd", "adam", "adamw", "rmsprop"]
for d in data_list:
    for opt in optimizers:
        for m in model_list:
            cfg = {
                "exp_name": f"exp5_opt_{m}_{d}_{opt}",
                "dataset": {"name": d, "imb_factor": 0.01},
            "model": {"name": m},
            "train": COMMON_TRAIN_CONFIG.copy()
            }
            cfg['train']['optimizer'] = opt

            # Auto-adjust learning rate
            if opt == "sgd":
                cfg['train']['lr'] = COMMON_TRAIN_CONFIG['lr']
            else:
                cfg['train']['lr'] = 0.001

            save_config(cfg, "exp5", cfg["exp_name"])

# =========================================================
# Experiment 6: Centroid drift
# =========================================================
# Count: 5 * 3 = 15
drift_factors = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
for d in data_list:
    for f in drift_factors:
            for m in model_list:
                cfg = {
                    "exp_name": f"exp6_drift_{m}_{d}_{f}",
                    "dataset": {"name": d, "imb_factor": f},
                    "model": {"name": m},
                "train": COMMON_TRAIN_CONFIG.copy()
                }
                save_config(cfg, "exp6", cfg["exp_name"])

# =========================================================
# Experiment 7: Batch size robustness
# =========================================================
# Count: 5 * 3 = 15
batch_sizes = [32, 64, 128, 512]
for d in data_list:
    for bs in batch_sizes:
        for m in model_list:
            cfg = {
                "exp_name": f"exp7_bs_{m}_{d}_{bs}",
                "dataset": {"name": d, "imb_factor": 0.01},
                "model": {"name": m},
                "train": COMMON_TRAIN_CONFIG.copy()
                }
            cfg['train']['batch_size'] = bs
            cfg['train']['lr'] = COMMON_TRAIN_CONFIG['lr'] * (bs / 256.0)
            save_config(cfg, "exp7", cfg["exp_name"])


# =========================================================
# Experiment 8: ViT ablation study
# =========================================================
# Count: 2
vit_variants = ["vit_tiny", "vit_tiny_orig",'EVA-02','EVA-02_orig']
for d in data_list:
    for m in vit_variants:
        cfg = {
            "exp_name": f"exp8_ablation_{m}_{d}",
            "dataset": {"name": d, "imb_factor": 0.01},
            "model": {"name": m},
            "train": COMMON_TRAIN_CONFIG.copy()
        }
        cfg['train']['epochs'] = COMMON_TRAIN_CONFIG['epochs']
        save_config(cfg, "exp8", cfg["exp_name"])
        

imagenet_models = ["vgg11", "resnet18"]
for m in imagenet_models:
    cfg = {
        "exp_name": f"exp10_imagenet_lt_{m}",
        "dataset": {"name": "imagenet_lt", "imb_factor": -1},
        "model": {"name": m},
        "train": COMMON_TRAIN_CONFIG.copy()
    }
    # ImageNet-specific adjustments
    cfg['train']['analyze_freq'] = 50

    cfg['train']['batch_size'] = 32
    # Linear scaling rule: 256 -> 0.1, 32 -> 0.0125
    cfg['train']['lr'] = 0.0125
    save_config(cfg, "exp10", cfg["exp_name"])
# =========================================================
# Statistics
# =========================================================
total_count = (
    len(datasets) * len(models) +
    (len(imb_factors) * len(model_list) +
    len(wd_list) * len(model_list) +
    len(optimizers) * len(model_list) +
    len(losses) * len(model_list) +
    len(drift_factors) * len(model_list) +
    len(batch_sizes) * len(model_list) +
    len(vit_variants)) * len(data_list)
    + len(data_list) * len(model_list) * 5
)

print("Successfully generated 8 groups of experiments.")
print(f"   - Total configurations: {total_count}")
print(f"   - Default Optimizer: {COMMON_TRAIN_CONFIG['optimizer']} ( lr={COMMON_TRAIN_CONFIG['lr']})")
print(f"   - Default Weight Decay: {COMMON_TRAIN_CONFIG['weight_decay']}")
print(f"   - Analysis Frequency: Every {COMMON_TRAIN_CONFIG['analyze_freq']} Epochs and Total epoch={COMMON_TRAIN_CONFIG['epochs']} ")