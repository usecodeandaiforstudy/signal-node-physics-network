from setuptools import setup, find_packages

# `long_description` is deliberately NOT read from README.md: that file opens with a status
# box listing six retracted claims, and a package description that ships a retraction table is
# worse than a short honest sentence. The full record is in FINDINGS.md.
DESCRIPTION = (
    "A neural network whose nodes are movable particles in a physical field, trained by "
    "operators that read the device rather than by gradients. Toy scale; simulated devices; "
    "see FINDINGS.md, which is the current statement of what this establishes."
)

setup(
    name="signal-node-physics-network",
    version="0.1.0",
    description=DESCRIPTION,
    # The repository's commits carry only a GitHub noreply address, so no author name is
    # recorded anywhere in it and none is invented here.
    author="usecodeandaiforstudy",
    url="https://github.com/usecodeandaiforstudy/signal-node-physics-network",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9",
        "numpy>=1.20",
        "matplotlib>=3.4",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
    ],
    # Aligned with the CI matrix (3.9 / 3.10 / 3.11) and with the README's Python badge. It
    # said ">=3.8", which is end-of-life and which CI has never tested.
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
