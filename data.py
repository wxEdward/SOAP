from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass
import json, os, random
import pandas as pd
from datasets import load_dataset as hf_load

@dataclass
class Example:
    id: str
    dialogue: str
    gold: Optional[str] = None  # optional gold SOAP text

def _iter_local_jsonl(path: str) -> Iterable[Example]:
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            yield Example(
                id=str(row.get('id', i)),
                dialogue=row['dialogue'],
                gold=row.get('soap', None)
            )

def _from_hf_omi(split: str) -> Iterable[Example]:
    ds = hf_load("omi-health/medical-dialogue-to-soap-summary", split=split)
    for i, r in enumerate(ds):
        # Fields per dataset card
        dial = r.get('dialogue') or r.get('conversation') or ""
        gold = r.get('summary') or r.get('soap') or r.get('soap_summary')
        yield Example(id=str(r.get('id', i)), dialogue=dial, gold=gold)

def _from_hf_meddialog(split: str) -> Iterable[Example]:
    ds = hf_load("bigbio/meddialog", name="meddialog_en", split=split)
    for i, r in enumerate(ds):
        # No gold SOAP; gold remains None
        dial = r.get('dialogue') or r.get('content') or ""
        yield Example(id=str(r.get('id', i)), dialogue=dial, gold=None)

def load_dataset(spec: str, split: str = "validation", limit: Optional[int] = None) -> List[Example]:
    """Load dataset given a spec:
    - 'local:/abs/path/to.jsonl'  (expects fields: id?, dialogue, soap?)
    - 'hf:omi-health/medical-dialogue-to-soap-summary'
    - 'hf:bigbio/meddialog'
    """
    if spec.startswith('local:'):
        path = spec.split(':',1)[1]
        data = list(_iter_local_jsonl(path))
    elif spec == 'hf:omi-health/medical-dialogue-to-soap-summary':
        data = list(_from_hf_omi(split))
    elif spec == 'hf:bigbio/meddialog':
        data = list(_from_hf_meddialog(split))
    else:
        raise ValueError(f'Unknown dataset spec: {spec}')
    if limit is not None:
        data = data[:limit]
    return data
