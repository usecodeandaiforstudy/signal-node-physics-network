"""
RE-EXPORT of `forward_training.spsa`.  This file used to be a byte-identical copy.

WHY THIS DIRECTION AND NOT THE OTHER: `forward_training/` is the v1 tree and is the
copy that is actually IMPORTED -- by `experiments/ref2d/ref2d_04_spsa_training.py`
and `ref2d_05_himal_training.py` -- while NOTHING in the repository imported
`training.spsa` (checked with an import scan, not by eye).  The two were
byte-identical, so the duplication was real, and deleting either side would have
broken the other's importers.

So this forwards instead: the name stays importable from here and there is ONE
implementation.  `forward_training/` carries no `__init__.py` and resolves as a
namespace package, which is how the two ref2d scripts already import it.
"""

from forward_training.spsa import *      # noqa: F401,F403
from forward_training.spsa import SPSAOptimizer   # noqa: F401
