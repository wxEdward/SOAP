# SOAPGen Kit (Dialogue → Structured SOAP Notes)

A lightweight, end-to-end pipeline to **load medical dialogue datasets**, **query an LLM** to produce **S/O/A/P sections**, and **evaluate** results (ROUGE, BERTScore, and section-wise checks).

> Built with inspiration from open projects like **MediSOAP** and **clinical visit note summarization** pipelines; this kit is self-contained for quick experiments.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or conda create -n soapgen python=3.10
pip install -r requirements.txt
export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY=...
python -m soapgen.run --help
```

Minimal example (uses the tiny sample in `data/sample.jsonl`):
```bash
python -m soapgen.run   --dataset local:/mnt/data/soapgen_kit/data/sample.jsonl   --provider openai --model gpt-4o-mini   --out_dir /mnt/data/soapgen_kit/outputs
```

To use a public dataset on Hugging Face (e.g., *omi-health/medical-dialogue-to-soap-summary*):
```bash
python -m soapgen.run   --dataset hf:omi-health/medical-dialogue-to-soap-summary   --provider openai --model gpt-4o-mini   --split validation --limit 50   --out_dir /mnt/data/soapgen_kit/outputs
```

## Datasets

- **omi-health/medical-dialogue-to-soap-summary** (synthetic doctor–patient dialogues with SOAP summaries)
- **bigbio/meddialog** (doctor–patient dialogues; no SOAP, but can be used for weak supervision or prompting)

## Evaluation

- ROUGE-L
- BERTScore (DeBERTa variant by default)
- Section coverage + formatting checks
