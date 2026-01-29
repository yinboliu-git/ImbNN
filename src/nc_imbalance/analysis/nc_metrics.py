"""Neural Collapse metrics computation and analysis."""
import torch
import numpy as np
import json
import os
from ..models.architectures import get_feature

class NCAnalyzer:
    def __init__(self, model, loader, device, save_dir, model_name, analyze_layers=False, save_logits=False):
        self.model = model
        self.loader = loader
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.analyze_layers = analyze_layers
        self.save_logits = save_logits

        # Hook storage
        self.layer_features = {}
        self.hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            # Global Average Pooling for spatial features if needed
            if output.dim() == 4:
                self.layer_features[name] = output.mean(dim=[2, 3]).detach()
            else:
                self.layer_features[name] = output.detach()
        return hook

    def register_hooks(self):
        # Only for ResNet structure
        if 'resnet' in self.model_name:
            # Monitor 4 stages and final layer (avgpool)
            layers_to_hook = {
                'layer1': self.model.layer1,
                'layer2': self.model.layer2,
                'layer3': self.model.layer3,
                'layer4': self.model.layer4,
                'avgpool': self.model.avgpool
            }
            for name, module in layers_to_hook.items():
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.layer_features = {}

    def compute_all_metrics(self, epoch):
        self.model.eval()

        # Storage containers
        all_feats = []
        all_labels = []
        all_preds = []
        all_logits = []

        # Layer-wise drift storage
        layer_drift_accum = {k: 0.0 for k in ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']} if self.analyze_layers else {}
        sample_count = 0

        if self.analyze_layers:
            self.register_hooks()

        with torch.no_grad():
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device)

                # Forward pass (triggers hooks)
                outputs = self.model(inputs)

                # 1. Basic data collection
                _, predicted = torch.max(outputs, 1)

                # Extract penultimate layer features
                feats = get_feature(self.model, inputs, self.model_name)

                all_feats.append(feats.cpu())
                all_labels.append(targets.cpu())
                all_preds.append(predicted.cpu())

                if self.save_logits:
                    all_logits.append(outputs.cpu())

                # 2. Layer-wise drift online computation (avoid memory explosion)
                if self.analyze_layers:
                    for name, feat_batch in self.layer_features.items():
                        # Accumulate batch features
                        # Drift = || Global Mean ||
                        # Accumulate sum(feat), divide by N later
                        if name not in layer_drift_accum: layer_drift_accum[name] = 0

                        # Accumulate batch feature sum (dimension [D])
                        if isinstance(layer_drift_accum[name], float):
                            layer_drift_accum[name] = feat_batch.sum(dim=0).cpu()
                        else:
                            layer_drift_accum[name] += feat_batch.sum(dim=0).cpu()

                    sample_count += inputs.size(0)

        if self.analyze_layers:
            self.remove_hooks()

        # Concatenate
        H = torch.cat(all_feats, dim=0)
        Y = torch.cat(all_labels, dim=0)
        P = torch.cat(all_preds, dim=0)

        # Compute main drift
        mu_G = H.mean(dim=0)
        drift_norm = torch.norm(mu_G).item()

        # Compute accuracy
        correct = (P == Y).float()
        overall_acc = correct.mean().item()

        # Compute tail accuracy (bottom 20% classes by sample count)
        # Get class counts
        unique, counts = torch.unique(Y, return_counts=True)
        # Sort to find tail classes
        sorted_indices = torch.argsort(counts)  # ascending order
        num_tail_cls = max(1, int(len(unique) * 0.2))  # bottom 20%
        tail_classes = unique[sorted_indices[:num_tail_cls]]

        mask_tail = torch.isin(Y, tail_classes.cpu())
        if mask_tail.sum() > 0:
            tail_acc = correct[mask_tail].mean().item()
        else:
            tail_acc = 0.0

        results = {
            "epoch": epoch,
            "overall_acc": overall_acc,
            "tail_acc": tail_acc,
            "drift_norm": drift_norm,
        }

        # Process layer-wise drift results
        if self.analyze_layers and sample_count > 0:
            layer_drift_results = {}
            for name, sum_vec in layer_drift_accum.items():
                global_mean = sum_vec / sample_count
                layer_drift_results[name] = torch.norm(global_mean).item()
            results["layer_wise_drift"] = layer_drift_results

        # Save JSON
        with open(os.path.join(self.save_dir, f"metrics_epoch_{epoch}.json"), 'w') as f:
            json.dump(results, f, indent=4)

        # Save tensors (including logits for post-hoc analysis)
        save_dict = {
            'predictions': P,
            'targets': Y,
            'drift_vector': mu_G
        }
        if self.save_logits:
             save_dict['logits'] = torch.cat(all_logits, dim=0)

        torch.save(save_dict, os.path.join(self.save_dir, f"raw_geo_epoch_{epoch}.pt"))

        print(f"Metrics: Acc={overall_acc:.4f} | TailAcc={tail_acc:.4f} | Drift={drift_norm:.4f}")