"""Generate paper-ready SVG figures from the current checked-in results.

The script is intentionally dependency-free so it can run in a minimal Python
environment. It writes SVG files under paper/figures/.
"""

from __future__ import annotations

from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "figures"


def svg_page(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #172026; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .label {{ font-size: 13px; }}
    .small {{ font-size: 11px; fill: #46535c; }}
    .axis {{ stroke: #8a979f; stroke-width: 1; }}
    .grid {{ stroke: #d9e0e4; stroke-width: 1; }}
    .pt {{ fill: #2878b5; }}
    .rmt {{ fill: #c65d35; }}
    .rec {{ fill: #6b6f76; }}
    .base {{ fill: #2f9c68; }}
  </style>
{body}
</svg>
"""


def bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" />'


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "start") -> str:
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{value}</text>'


def figure_real_lm() -> None:
    data = [
        ("TinyStories", "Baseline", 1.5499, "base"),
        ("TinyStories", "Per-token", 1.5802, "pt"),
        ("TinyStories", "Recurrent", 1.7029, "rec"),
        ("WikiText-103", "Baseline", 3.6202, "base"),
        ("WikiText-103", "Per-token", 3.6831, "pt"),
        ("WikiText-103", "Recurrent", 3.7861, "rec"),
    ]
    width, height = 860, 430
    chart_x, chart_y = 190, 70
    chart_w, chart_h = 580, 280
    max_val = 4.0
    body = [text(40, 36, "35M Real-Data Validation Loss", "title")]
    for i in range(5):
        y = chart_y + chart_h - i * chart_h / 4
        val = i * max_val / 4
        body.append(f'<line class="grid" x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" />')
        body.append(text(chart_x - 10, y + 4, f"{val:.1f}", "small", "end"))
    body.append(f'<line class="axis" x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" />')
    body.append(f'<line class="axis" x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" />')

    group_x = {"TinyStories": chart_x + 35, "WikiText-103": chart_x + 320}
    offsets = {"Baseline": 0, "Per-token": 52, "Recurrent": 104}
    for dataset, model, val, cls in data:
        x = group_x[dataset] + offsets[model]
        h = val / max_val * chart_h
        y = chart_y + chart_h - h
        body.append(bar(x, y, 38, h, cls))
        body.append(text(x + 19, y - 8, f"{val:.3f}", "small", "middle"))
    body.append(text(group_x["TinyStories"] + 70, chart_y + chart_h + 36, "TinyStories", "label", "middle"))
    body.append(text(group_x["WikiText-103"] + 70, chart_y + chart_h + 36, "WikiText-103", "label", "middle"))
    body.append(text(610, 55, "Baseline", "small"))
    body.append(bar(575, 44, 24, 14, "base"))
    body.append(text(610, 78, "Per-token", "small"))
    body.append(bar(575, 67, 24, 14, "pt"))
    body.append(text(610, 101, "Recurrent 128x2048", "small"))
    body.append(bar(575, 90, 24, 14, "rec"))
    body.append(text(40, 390, "Lower is better. Recurrent row uses the fairer 128x2048 shape.", "small"))
    (OUT_DIR / "real_lm_loss.svg").write_text(svg_page(width, height, "\n".join(body)), encoding="utf-8")


def figure_rmt_vs_per_token() -> None:
    data = [
        ("Copy-32", 1.000, 0.680, 0.491),
        ("Copy-64", 1.000, 0.031, 0.000),
        ("Needle-32", 1.000, 0.702, 0.225),
        ("Needle-64", 1.000, 0.877, 0.104),
        ("KV-16", 0.168, 0.098, 0.053),
    ]
    width, height = 920, 460
    chart_x, chart_y = 90, 70
    chart_w, chart_h = 760, 300
    body = [text(40, 36, "Direct Per-Token vs RMT Synthetic Accuracy", "title")]
    for i in range(6):
        y = chart_y + chart_h - i * chart_h / 5
        val = i / 5
        body.append(f'<line class="grid" x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" />')
        body.append(text(chart_x - 10, y + 4, f"{val:.1f}", "small", "end"))
    body.append(f'<line class="axis" x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" />')
    body.append(f'<line class="axis" x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" />')
    step = chart_w / len(data)
    for idx, (task, pt_val, rmt_val, rmt_std) in enumerate(data):
        base_x = chart_x + idx * step + 26
        pt_h = pt_val * chart_h
        rmt_h = rmt_val * chart_h
        body.append(bar(base_x, chart_y + chart_h - pt_h, 34, pt_h, "pt"))
        body.append(bar(base_x + 42, chart_y + chart_h - rmt_h, 34, rmt_h, "rmt"))
        if rmt_std > 0:
            cx = base_x + 59
            y1 = chart_y + chart_h - min(1.0, rmt_val + rmt_std) * chart_h
            y2 = chart_y + chart_h - max(0.0, rmt_val - rmt_std) * chart_h
            body.append(f'<line x1="{cx:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{y2:.1f}" stroke="#172026" stroke-width="1.5" />')
            body.append(f'<line x1="{cx - 6:.1f}" y1="{y1:.1f}" x2="{cx + 6:.1f}" y2="{y1:.1f}" stroke="#172026" stroke-width="1.5" />')
            body.append(f'<line x1="{cx - 6:.1f}" y1="{y2:.1f}" x2="{cx + 6:.1f}" y2="{y2:.1f}" stroke="#172026" stroke-width="1.5" />')
        body.append(text(base_x + 38, chart_y + chart_h + 28, task, "small", "middle"))
    body.append(bar(650, 42, 24, 14, "pt"))
    body.append(text(684, 54, "Per-token", "small"))
    body.append(bar(650, 66, 24, 14, "rmt"))
    body.append(text(684, 78, "RMT-style", "small"))
    body.append(text(40, 415, "Bars show three-seed means; RMT error bars show standard deviation.", "small"))
    (OUT_DIR / "rmt_vs_per_token.svg").write_text(svg_page(width, height, "\n".join(body)), encoding="utf-8")


def figure_memory_layout() -> None:
    width, height = 920, 330
    body = [text(40, 36, "Memory Allocation Under Similar Budgets", "title")]
    panels = [
        ("Full KV", "many wide K/V states", 50, "base", 16),
        ("Per-token", "many compressed K/V states", 275, "pt", 16),
        ("Custom recurrent", "fixed summary slots", 500, "rec", 8),
        ("RMT-style", "memory tokens across chunks", 725, "rmt", 8),
    ]
    for title, subtitle, x, cls, count in panels:
        body.append(text(x, 82, title, "label"))
        body.append(text(x, 104, subtitle, "small"))
        for i in range(count):
            px = x + (i % 4) * 36
            py = 130 + (i // 4) * 30
            body.append(f'<rect class="{cls}" x="{px}" y="{py}" width="26" height="18" rx="3" opacity="0.9" />')
        if title == "Custom recurrent":
            body.append(text(x, 265, "compress chunks into slots", "small"))
        elif title == "RMT-style":
            body.append(text(x, 265, "read/write via memory tokens", "small"))
        else:
            body.append(text(x, 265, "indexed by token position", "small"))
    body.append(text(40, 306, "Same budget can mean very different storage geometry.", "small"))
    (OUT_DIR / "memory_layout.svg").write_text(svg_page(width, height, "\n".join(body)), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_real_lm()
    figure_rmt_vs_per_token()
    figure_memory_layout()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

