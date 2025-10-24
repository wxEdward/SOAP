from __future__ import annotations
import json, os
from pathlib import Path
from typing import Optional
import typer
from rich import print
from rich.progress import track
from .data import load_dataset
from .models import GenConfig, generate_soap
from .eval import evaluate

app = typer.Typer(add_help_option=True)

@app.command()
def main(
    dataset: str = typer.Option(..., help="Dataset spec: local:/abs/file.jsonl | hf:omi-health/medical-dialogue-to-soap-summary | hf:bigbio/meddialog"),
    split: str = typer.Option('validation', help="HF split (ignored for local)"),
    limit: Optional[int] = typer.Option(None, help="Limit number of examples"),
    provider: str = typer.Option('openai', help="Model provider: openai|anthropic"),
    model: str = typer.Option('gpt-4o-mini', help="Model id (e.g., gpt-4o-mini, o3-mini, claude-3-5-sonnet-latest)"),
    temperature: float = typer.Option(0.2),
    max_tokens: int = typer.Option(512),
    out_dir: str = typer.Option('./outputs', help="Where to write predictions + metrics")
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset, split=split, limit=limit)
    cfg = GenConfig(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    rows = []
    for ex in track(data, description="Generating"):
        pred = generate_soap(ex.dialogue, cfg)
        row = {
            'id': ex.id,
            'dialogue': ex.dialogue,
            'gold': ex.gold,
            'pred': pred
        }
        if ex.gold:
            row['metrics'] = evaluate(ex.gold, pred)
        rows.append(row)

    pred_path = out / 'predictions.jsonl'
    with open(pred_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"[green]Wrote {pred_path}[/green]")

    # Write aggregate metrics if gold available
    has_gold = [r for r in rows if r.get('metrics')]
    if has_gold:
        import numpy as np
        rouge = np.mean([r['metrics']['rougeL'] for r in has_gold])
        bertf = np.mean([r['metrics']['bertscore_F1'] for r in has_gold])
        coverage = {k: float(np.mean([r['metrics'][k] for r in has_gold])) for k in ['has_S','has_O','has_A','has_P']}
        agg = {'mean_rougeL': float(rouge), 'mean_bertscore_F1': float(bertf), **coverage, 'n_eval': len(has_gold)}
        agg_path = out / 'metrics.json'
        with open(agg_path, 'w', encoding='utf-8') as f:
            json.dump(agg, f, indent=2)
        print(f"[blue]Wrote {agg_path}[/blue]")
    else:
        print("[yellow]No gold references found — only predictions written.[/yellow]")

if __name__ == '__main__':
    app()
