import json
import math
from pathlib import Path

def parse_summary(log_path: Path):
    if not log_path.exists():
        return None
    
    summary = None
    last_metrics = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("event") == "metrics":
                last_metrics = data
            elif data.get("event") == "summary":
                summary = data
                
    if not summary and last_metrics:
        # Fallback if training interrupted but some steps finished
        summary = {"final_metrics": last_metrics}
    return summary

def main():
    logs_dir = Path("logs")
    configs = {
        "Baseline": "full_baseline.jsonl",
        "Per-Token (Many-Small)": "full_per_token.jsonl",
        "Recurrent (Few-Rich, mean_gru)": "full_recurrent_mean_gru.jsonl",
        "Recurrent (Few-Rich, cross_attn)": "full_recurrent_cross_attn.jsonl",
    }
    
    rows = []
    for name, filename in configs.items():
        summary = parse_summary(logs_dir / filename)
        if not summary:
            continue
        
        final_metrics = summary.get("final_metrics", {})
        val_loss = final_metrics.get("validation_loss", float("nan"))
        ppl = math.exp(val_loss) if not math.isnan(val_loss) else float("nan")
        tokens_per_sec = final_metrics.get("tokens_per_sec", 0.0)
        peak_vram = final_metrics.get("peak_gpu_memory_mb", 0.0)
        
        param_cnt = summary.get("parameter_count", {}).get("total", 0)
        mem_floats = summary.get("memory_budget", {}).get("floats", 0)
        
        rows.append({
            "Model": name,
            "Val Loss": f"{val_loss:.4f}",
            "Perplexity": f"{ppl:.2f}",
            "Tokens/sec": f"{tokens_per_sec/1000:.1f}k" if tokens_per_sec > 0 else "N/A",
            "Peak VRAM": f"{peak_vram:.2f} MB" if peak_vram > 0 else "N/A",
            "Params": f"{param_cnt:,}",
            "Mem Floats": f"{mem_floats:,}",
        })
        
    if not rows:
        print("No log files found in logs/ directory. Make sure you run the full comparison first.")
        return
        
    headers = ["Model", "Val Loss", "Perplexity", "Tokens/sec", "Peak VRAM", "Params", "Mem Floats"]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(row[h]))
            
    # Print markdown table
    header_str = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    sep_str = "-|-".join("-" * col_widths[h] for h in headers)
    print(header_str)
    print(sep_str)
    for row in rows:
        print(" | ".join(f"{row[h]:<{col_widths[h]}}" for h in headers))

if __name__ == "__main__":
    main()
