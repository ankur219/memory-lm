"""Run the first published recurrent baseline probe.

This script compares the existing recurrent baseline against an RMT-style
memory-token baseline on the same synthetic tasks. It intentionally starts at
the small synthetic scale before any real-data run, because RMT-style memory
adds memory tokens to the attention sequence and can be compute-heavy.
"""

import subprocess
import sys


COMMANDS = [
    [
        sys.executable,
        "experiments/run_copy_sweep.py",
        "--lengths",
        "16",
        "32",
        "64",
        "--models",
        "recurrent",
        "rmt",
        "--recurrent-shapes",
        "256x128",
        "--steps",
        "3000",
        "--num-examples",
        "30000",
        "--test-examples",
        "3000",
        "--batch-size",
        "128",
        "--csv-path",
        "logs/copy_rmt_baseline.csv",
    ],
    [
        sys.executable,
        "experiments/run_needle_sweep.py",
        "--gaps",
        "16",
        "32",
        "64",
        "--models",
        "recurrent",
        "rmt",
        "--recurrent-shapes",
        "256x128",
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
        "logs/needle_rmt_baseline.csv",
    ],
    [
        sys.executable,
        "experiments/run_kv_sweep.py",
        "--pairs",
        "4",
        "8",
        "16",
        "--models",
        "recurrent",
        "rmt",
        "--recurrent-shapes",
        "256x128",
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
        "logs/kv_rmt_baseline.csv",
    ],
]


if __name__ == "__main__":
    for command in COMMANDS:
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
