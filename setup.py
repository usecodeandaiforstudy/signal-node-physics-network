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
    # THE FLOOR IS 3.11, AND IT IS ENFORCED BY THE ONLY VERSION CI RUNS.
    #
    # It said ">=3.9" while CI tested 3.9, 3.10 and 3.11 -- and the 3.9 job kept
    # failing at import on `np.ndarray | None`, an annotation evaluated at module
    # level that is a TypeError before 3.10.  That is not a defect in the code; it
    # is a statement about which version the code is written for, and the matrix
    # spent three jobs' minutes saying it three times.
    #
    # So it is said once, here, and CI runs one interpreter that matches it.  A
    # single tested version that is also the declared requirement is honest; a
    # matrix that mostly reports syntax differences is not free.
    python_requires=">=3.11",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
