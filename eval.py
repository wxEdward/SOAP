from __future__ import annotations
from typing import Dict, Tuple
import re
import numpy as np
from rouge_score import rouge_scorer
import bert_score

SOAP_RE = re.compile(r'^(S|O|A|P)\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)

def split_soap(text: str) -> Dict[str,str]:
    parts = {'S':'', 'O':'', 'A':'', 'P':''}
    for m in SOAP_RE.finditer(text or ''):
        key = m.group(1).upper()
        val = m.group(2).strip()
        parts[key] = val
    return parts

def coverage_ok(text: str) -> Dict[str, bool]:
    parts = split_soap(text)
    return {k: bool(parts[k]) for k in ['S','O','A','P']}

def rougeL(gold: str, pred: str) -> float:
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return sc.score(gold or '', pred or '')['rougeL'].fmeasure

def bertscore(gold: str, pred: str, lang: str = 'en') -> float:
    P, R, F = bert_score.score([pred or ''], [gold or ''], lang=lang, verbose=False)
    return float(F[0].item())

def evaluate(gold: str, pred: str) -> Dict[str, float | bool]:
    cov = coverage_ok(pred)
    scores = {
        'rougeL': rougeL(gold or '', pred or ''),
        'bertscore_F1': bertscore(gold or '', pred or '')
    }
    return {**scores, **{f'has_{k}': v for k,v in cov.items()}}
