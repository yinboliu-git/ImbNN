import os
import sys
from datasets import ImbalancedDatasetGenerator

def check_all_datasets():
    # All dataset names to test
    dataset_names = ['mnist', 'fmnist', 'cifar10', 'cifar100']

    print(f"Checking datasets in: {os.path.abspath('./datasets')}")
    print("="*60)

    all_passed = True

    for name in dataset_names:
        print(f"Checking [{name}]...", end=" ")
        try:
            # Try to initialize and get loader (triggers data check)
            # Use f=1.0 (balanced) to test reading all data
            data_gen = ImbalancedDatasetGenerator(name=name, root='./datasets', imb_factor=1.0)
            loader = data_gen.get_loader(batch_size=4)

            # Try reading one batch to ensure files are not corrupted
            first_batch = next(iter(loader))

            print(f"OK! (Samples: {len(data_gen.indices)})")

        except RuntimeError as e:
            print(f"Failed!")
            print(f"   Reason: Data not found or structure incorrect.")
            print(f"   Error Details: {e}")
            all_passed = False
        except Exception as e:
            print(f"Error!")
            print(f"   Error Details: {e}")
            all_passed = False

    print("="*60)
    if all_passed:
        print("Congratulations! All datasets are ready.")
        print("You can run 'bash run.sh' now.")
    else:
        print("Some datasets failed. Please fix the file locations above.")

if __name__ == "__main__":
    check_all_datasets()