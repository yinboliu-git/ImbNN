import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import glob
from PIL import Image

# Try importing datasets library (required for Parquet files)
try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None

class HFDatasetWrapper(torch.utils.data.Dataset):
    """Wraps Hugging Face Dataset (Parquet) as PyTorch Dataset"""
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item['image'] 
        label = item['label']

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        return img, label

class ImbalancedDatasetGenerator:
    def __init__(self, name, root='./datasets', imb_type='exp', imb_factor=0.01):
        self.name = name
        self.root = root
        if not os.path.exists(root): os.makedirs(root)
        
        # ==========================================
        # 1. Define common transforms
        # ==========================================
        base_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # ==========================================
        # 2. Load dataset
        # ==========================================
        self.dataset = None
        self.targets = None

        if name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
            if name == 'mnist':
                full_tf = transforms.Compose([transforms.Grayscale(3), base_transform])
                self.dataset = datasets.MNIST(root, train=True, download=True, transform=full_tf)
            elif name == 'fmnist':
                full_tf = transforms.Compose([transforms.Grayscale(3), base_transform])
                self.dataset = datasets.FashionMNIST(root, train=True, download=True, transform=full_tf)
            elif name == 'cifar10':
                self.dataset = datasets.CIFAR10(root, train=True, download=True, transform=base_transform)
            elif name == 'cifar100':
                self.dataset = datasets.CIFAR100(root, train=True, download=True, transform=base_transform)
            elif name == 'svhn':
                svhn_root = os.path.join(root, 'svhn')
                if not os.path.exists(svhn_root): os.makedirs(svhn_root, exist_ok=True)
                self.dataset = datasets.SVHN(svhn_root, split='train', download=True, transform=base_transform)

            # Get targets
            if hasattr(self.dataset, 'targets'):
                self.targets = np.array(self.dataset.targets)
            elif hasattr(self.dataset, 'labels'):
                self.targets = np.array(self.dataset.labels)

        # --- ImageNet-LT ---
        elif name == 'imagenet_lt':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            
            possible_paths = [
                os.path.join(root, 'imagenet', 'train'),
                os.path.join(root, 'train'),
                root
            ]
            data_dir = None
            for p in possible_paths:
                if os.path.exists(p):
                    if len(glob.glob(os.path.join(p, "*.parquet"))) > 0:
                        data_dir = p
                        break
                    try:
                        if any(os.path.isdir(os.path.join(p, d)) for d in os.listdir(p)):
                            data_dir = p
                            break
                    except:
                        pass
            
            if not data_dir:
                 raise FileNotFoundError(f"Cannot find ImageNet data in {possible_paths}.")

            print(f"Found ImageNet data at: {data_dir}")

            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
            
            if len(parquet_files) > 0:
                if hf_datasets is None:
                    raise ImportError("Found .parquet files but 'datasets' missing.")
                
                print(f"Detected {len(parquet_files)} parquet files.")
                hf_ds = hf_datasets.load_dataset("parquet", data_files={'train': parquet_files}, split='train')

                self.dataset = HFDatasetWrapper(hf_ds, transform=train_transform)
                self.num_classes = 1000
                print("Extracting targets...")
                self.targets = np.array(hf_ds['label'])
            else:
                print("Detected standard folder structure.")
                self.dataset = datasets.ImageFolder(data_dir, transform=train_transform)
                self.num_classes = 1000
                self.targets = np.array(self.dataset.targets)

        else:
            raise ValueError(f"Dataset {name} not supported.")

        if self.targets is None:
             raise ValueError("Failed to load targets.")
             
        self.num_classes = len(np.unique(self.targets))

        # 3. Compute indices
        self.indices = self._get_imbalanced_indices(imb_type, imb_factor)

        # ==========================================
        # 4. Compute and save img_num_list
        # ==========================================
        # Derive actual class counts from selected indices
        # This ensures img_num_list is accurate regardless of imb_factor
        subset_targets = self.targets[self.indices]
        self.img_num_list = [0] * self.num_classes
        unique_labels, counts = np.unique(subset_targets, return_counts=True)

        for label, count in zip(unique_labels, counts):
            if label < self.num_classes:
                self.img_num_list[int(label)] = int(count)

        print(f"Dataset initialized. Class counts (first 10): {self.img_num_list[:10]}...")

    def _get_imbalanced_indices(self, imb_type, imb_factor):
        # ImageNet-LT native distribution (imb_factor < 0)
        if imb_factor < 0:
            return list(range(len(self.targets)))

        img_num_per_cls = []
        n_max = len(self.targets) / self.num_classes
        for cls_idx in range(self.num_classes):
            if imb_type == 'exp':
                num = n_max * (imb_factor**(cls_idx / (self.num_classes - 1.0)))
                img_num_per_cls.append(int(num))
            else:
                img_num_per_cls.append(int(n_max))

        new_indices = []
        classes = np.unique(self.targets)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(self.targets == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_indices.extend(selec_idx)
        
        return new_indices

    def get_loader(self, batch_size=128, num_workers=4):
        print(f'Loading {self.name} with {len(self.indices)} samples...')
        subset = Subset(self.dataset, self.indices)
        return DataLoader(
            subset, 
            batch_size=batch_size, 
            pin_memory=True, 
            shuffle=True, 
            num_workers=num_workers
        )