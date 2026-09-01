"""Run the KVM-style external baseline probe on synthetic tasks.

KVM stores layerwise compressed keys and values, so the matched 32,768-float
synthetic shape is 128 slots x 64 dims:

    2 layers x 128 slots x 2(K,V) x 64 dims = 32,768 floats

This runner launches only `kvm`. The existing custom recurrent matched-budget
shape is `256x128`, so putting both models in one command would force one of
them onto an unfair shape. Compare these logs against the existing recurrent,
RMT, and per-token tables after the run.
"""

import subprocess
import sys


COMMANDS = [
    [
        sys.executable,
        "experiments/run_copy_sweep.py",
        "--lengths",
        "32",
        "64",
        "128",
        "--models",
        "kvm",
        "--recurrent-shapes",
        "128x64",
        "--recurrent-learned-initial",
        "--steps",
        "3000",
        "--num-examples",
        "30000",
        "--test-examples",
        "3000",
        "--batch-size",
        "128",
        "--csv-path",
        "logs/copy_kvm_baseline.csv",
    ],
    [
        sys.executable,
        "experiments/run_needle_sweep.py",
        "--gaps",
        "32",
        "64",
        "128",
        "--models",
        "kvm",
        "--recurrent-shapes",
        "128x64",
        "--recurrent-learned-initial",
        "--steps",
        "3000",
        "--num-examples",
        "30000",
        "--test-examples",
        "3000",
        "--batch-size",
        "128",
        "--answer-loss-weight",
        "10",
        "--csv-path",
        "logs/needle_kvm_baseline.csv",
    ],
    [
        sys.executable,
        "experiments/run_kv_sweep.py",
        "--pairs",
        "4",
        "8",
        "16",
        "--models",
        "kvm",
        "--recurrent-shapes",
        "128x64",
        "--recurrent-learned-initial",
        "--steps",
        "5000",
        "--num-examples",
        "50000",
        "--test-examples",
        "5000",
        "--batch-size",
        "128",
        "--num-keys",
        "16",
        "--num-values",
        "16",
        "--answer-loss-weight",
        "20",
        "--value-mode",
        "random",
        "--csv-path",
        "logs/kv_kvm_baseline.csv",
    ],
]


if __name__ == "__main__":
    for command in COMMANDS:
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
