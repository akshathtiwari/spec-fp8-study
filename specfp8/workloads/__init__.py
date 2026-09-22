"""Workload definitions — prompt sources and validators."""

from specfp8.workloads.gsm8k import Gsm8kWorkload
from specfp8.workloads.mtbench import MtBenchWorkload

WORKLOADS = {
    "gsm8k": Gsm8kWorkload,
    "mtbench": MtBenchWorkload,
}


def get_workload(name: str):
    """Instantiate a workload by name."""
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload {name!r}. Have: {sorted(WORKLOADS)}")
    return WORKLOADS[name]()
