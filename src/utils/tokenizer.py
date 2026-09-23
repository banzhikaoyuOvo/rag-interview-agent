"""统一分词：英文技术词整体保留，中文走 jieba"""
from __future__ import annotations

import re

import jieba

_EN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    last_end = 0

    for m in _EN_PATTERN.finditer(text):
        if m.start() > last_end:
            zh = text[last_end : m.start()]
            tokens.extend(t.strip() for t in jieba.cut(zh) if t.strip())
        tokens.append(m.group().lower())
        last_end = m.end()

    if last_end < len(text):
        zh = text[last_end:]
        tokens.extend(t.strip() for t in jieba.cut(zh) if t.strip())

    return [t for t in tokens if t]