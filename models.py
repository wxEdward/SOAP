from __future__ import annotations
from typing import Dict, Optional
import os, time
from dataclasses import dataclass
from .prompts import render_prompt

@dataclass
class GenConfig:
    provider: str  # 'openai' | 'anthropic'
    model: str
    temperature: float = 0.2
    max_tokens: int = 512

def _call_openai(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content": system},
            {"role":"user","content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def _call_anthropic(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role":"user","content": user}],
    )
    # Anthropic SDK returns content blocks
    blocks = msg.content
    txt = ''.join(b.text for b in blocks if getattr(b, 'type', 'text') == 'text' or hasattr(b, 'text'))
    return txt

def generate_soap(dialogue: str, cfg: GenConfig) -> str:
    pr = render_prompt(dialogue)
    if cfg.provider == 'openai':
        return _call_openai(cfg.model, pr['system'], pr['user'], cfg.temperature, cfg.max_tokens)
    elif cfg.provider == 'anthropic':
        return _call_anthropic(cfg.model, pr['system'], pr['user'], cfg.temperature, cfg.max_tokens)
    else:
        raise ValueError('Unsupported provider: ' + cfg.provider)
