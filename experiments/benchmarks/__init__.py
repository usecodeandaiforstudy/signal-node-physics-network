"""
V3 Benchmark Infrastructure

Standard dataset loading, sklearn baseline comparison,
and report generation for producing citable benchmark numbers.
"""

from .datasets import load_dataset, list_datasets
from .baselines import run_sklearn_baselines
from .report import generate_table
from .run_benchmark import run_benchmark
