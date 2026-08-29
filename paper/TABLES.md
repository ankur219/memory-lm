# Paper Tables

## Main Real-Data Table

| Dataset | Model | Val Loss | Perplexity | Params | Mem Floats |
|---|---|---:|---:|---:|---:|
| TinyStories | Baseline | 1.5499 | 4.71 | 38,179,584 | 786,432 |
| TinyStories | Per-token | 1.5802 | 4.86 | 35,431,680 | 262,144 |
| TinyStories | Recurrent `8x32768` | 1.7068 | 5.51 | 35,431,680 | 262,144 |
| TinyStories | Recurrent `128x2048` | 1.7029 | 5.49 | 35,450,112 | 262,144 |
| WikiText-103 | Baseline | 3.6202 | 37.35 | 38,179,584 | 786,432 |
| WikiText-103 | Per-token | 3.6831 | 39.77 | 35,431,680 | 262,144 |
| WikiText-103 | Recurrent `8x32768` | 3.7973 | 44.58 | 35,431,680 | 262,144 |
| WikiText-103 | Recurrent `128x2048` | 3.7861 | 44.08 | 35,450,112 | 262,144 |

## Direct Per-Token vs RMT Synthetic Table

| Task | Model | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| Copy-32 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Copy-32 | RMT-style | 0.112 | 0.963 | 0.964 | 0.680 | 0.491 |
| Copy-64 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Copy-64 | RMT-style | 0.031 | 0.030 | 0.031 | 0.031 | 0.000 |
| Needle-32 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Needle-32 | RMT-style | 0.938 | 0.677 | 0.491 | 0.702 | 0.225 |
| Needle-64 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Needle-64 | RMT-style | 0.903 | 0.966 | 0.763 | 0.877 | 0.104 |
| Random KV-16 | Per-token | 0.163 | 0.167 | 0.173 | 0.168 | 0.005 |
| Random KV-16 | RMT-style | 0.067 | 0.159 | 0.067 | 0.098 | 0.053 |

## Recurrent Shape Check

| Dataset | Shape | Val Loss | Delta vs `8x32768` | Mem Floats |
|---|---|---:|---:|---:|
| TinyStories | `8x32768` | 1.7068 | - | 262,144 |
| TinyStories | `128x2048` | 1.7029 | -0.0039 | 262,144 |
| WikiText-103 | `8x32768` | 3.7973 | - | 262,144 |
| WikiText-103 | `128x2048` | 3.7861 | -0.0112 | 262,144 |

