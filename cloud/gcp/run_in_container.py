"""Run `boot_variance` on a plain GPU host, outside Modal, unchanged.

F042's decisive test is an intervention: install a chosen NVIDIA driver on
one L4 and measure, then swap the driver on the SAME card and measure again.
Modal cannot do it -- the platform supplies the driver -- so it runs on a
GCP VM. The result only means something if the measurement procedure is the
one that produced every earlier session, so this file changes none of it.

It imports `cloud/modal_probe.py` and calls the `boot_variance` function body
directly. The only Modal object that body touches at run time is the results
volume's `commit()`, called after each record; it is replaced with a no-op,
and records land in /results, a directory mounted from the VM's disk so they
survive the reboot that swaps the driver.

    python cloud/gcp/run_in_container.py --mechanism dflash,none --boots 12 --noasync
"""
import argparse
import os
import sys
import types

REPO = "/root/spec-fp8-study"
sys.path.insert(0, os.path.join(REPO, "cloud"))
sys.path.insert(0, REPO)

import modal_probe  # noqa: E402  (declarative at import: no Modal auth or network)

modal_probe.results_vol = types.SimpleNamespace(commit=lambda: None)
os.makedirs("/results", exist_ok=True)
os.environ.setdefault("SPECFP8_PLATFORM", "gcp")

ap = argparse.ArgumentParser()
ap.add_argument("--mechanism", default="dflash,none")
ap.add_argument("--boots", type=int, default=12)
ap.add_argument("--noasync", action="store_true")
ap.add_argument("--sweep-file", default="sweeps/boot_stability.yaml")
a = ap.parse_args()

fn = modal_probe.boot_variance
raw = fn.get_raw_f() if hasattr(fn, "get_raw_f") else fn.local
print(raw(boots=a.boots, mechanism=a.mechanism,
          sweep_file=a.sweep_file, noasync=a.noasync))
