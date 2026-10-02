"""Multi-graph readout extension (design + the correctness anchor)."""

from .graphs import (                                    # noqa: F401
    EXT_ARCH_PREFIX,
    BASE_ARCH_KEYS,
    BASE_TASK_STRINGS,
    Graph,
    Linear,
    MultiGraphStack,
    Nonlinear,
    check_isolation,
    ext_arch_key,
    graphs_from_base,
)
