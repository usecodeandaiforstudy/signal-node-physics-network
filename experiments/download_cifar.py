"""
download_cifar.py — Download the CIFAR-10 dataset to the cifar/ directory at the project root.

Usage:
    python experiments/download_cifar.py

After download, the directory structure will be:
    cifar/
    ├── batches.meta
    ├── data_batch_1 ~ data_batch_5
    └── test_batch

If you have already downloaded CIFAR-10 manually, you can simply rename the
extracted cifar-10-batches-py directory to cifar and place it at the project root.
"""
import os
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(ROOT, "cifar")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def main():
    # Skip if the dataset already exists
    if os.path.exists(os.path.join(TARGET_DIR, "data_batch_1")):
        print(f"Dataset already exists at {TARGET_DIR}, skipping download.")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    tar_path = os.path.join(ROOT, "cifar-10-python.tar.gz")

    print("Downloading CIFAR-10 dataset...")
    print(f"  Source: {URL}")
    urllib.request.urlretrieve(URL, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(ROOT)

    # Move the contents of the extracted cifar-10-batches-py directory into cifar/
    extracted = os.path.join(ROOT, "cifar-10-batches-py")
    if os.path.exists(extracted):
        for name in os.listdir(extracted):
            src = os.path.join(extracted, name)
            dst = os.path.join(TARGET_DIR, name)
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
        os.rmdir(extracted)

    # Clean up the downloaded archive
    if os.path.exists(tar_path):
        os.remove(tar_path)

    print(f"Done! Dataset is ready at {TARGET_DIR}")


if __name__ == "__main__":
    main()
