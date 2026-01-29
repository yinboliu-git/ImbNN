"""Training script for neural collapse experiments."""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nc_imbalance.data import ImbalancedDatasetGenerator
from nc_imbalance.models import ModelFactory, get_feature
from nc_imbalance.analysis import NCAnalyzer
from nc_imbalance.training import LogitAdjustmentLoss, BalancedSoftmaxLoss, FocalLoss


def main():
    parser = argparse.ArgumentParser(description='Train models for neural collapse analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--save_root', type=str, default='./results', help='Root directory for saving results')
    args = parser.parse_args()

    # Check config path
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = config['exp_name']
    save_dir = os.path.join(args.save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Data loading
    data_gen = ImbalancedDatasetGenerator(
        name=config['dataset']['name'],
        imb_factor=config['dataset']['imb_factor']
    )
    train_loader = data_gen.get_loader(batch_size=config['train']['batch_size'])
    num_classes = data_gen.num_classes

    # Get class sample counts
    if hasattr(data_gen, 'img_num_list'):
     cls_num_list = data_gen.img_num_list
    else:
        raise AttributeError("data_gen missing 'img_num_list'. Please update data_loader.py.")

    # 2. Model
    model = ModelFactory.get_model(config['model']['name'], num_classes=num_classes)
    model = model.to(device)

    # 3. Loss function selection
    loss_type = config['train']['loss']
    drift_penalty_weight = config['train'].get('drift_penalty', 0.0)

    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss(gamma=2.0)
    elif loss_type == 'logit_adjustment':
        criterion = LogitAdjustmentLoss(cls_num_list, tau=1.0)
    elif loss_type == 'balanced_softmax':
        criterion = BalancedSoftmaxLoss(cls_num_list)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")

    # 4. Optimizer
    opt_type = config['train']['optimizer']
    lr = float(config['train']['lr'])
    wd = float(config['train']['weight_decay'])
    
    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

    # 5. Analyzer
    analyze_layers = config['train'].get('analyze_layers', False)
    save_logits = config['train'].get('save_logits', False)
    
    analyzer = NCAnalyzer(
        model, train_loader, device, save_dir, 
        config['model']['name'],
        analyze_layers=analyze_layers,
        save_logits=save_logits
    )

    # 6. Training loop
    epochs = config['train']['epochs']
    print(f"Starting {exp_name} | Loss: {loss_type} | DriftPenalty: {drift_penalty_weight}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if drift_penalty_weight > 0:
                feats = get_feature(model, x, config['model']['name'])
                logits = model.fc(feats)
                cls_loss = criterion(logits, y)
                mu_batch = feats.mean(dim=0)
                drift_loss = torch.norm(mu_batch, p=2) ** 2
                loss = cls_loss + drift_penalty_weight * drift_loss
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()

        if epoch == epochs or (epoch % config['train']['analyze_freq'] == 0):
            print(f"Epoch [{epoch}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
            analyzer.compute_all_metrics(epoch)

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"Training completed. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
