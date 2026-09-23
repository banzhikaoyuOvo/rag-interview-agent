"""文档解析：JSON / Markdown -> 统一 Chunk"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_id: str
    category: str
    visibility: str
    metadata_json: str

    def to_dict(self) -> dict:
        return asdict(self)


def _make_chunk_id(source_file: str, index: int, text: str) -> str:
    raw = f"{source_file}::{index}::{text[:128]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _join_fields(*values) -> str:
    parts: list[str] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, list):
            parts.append("、".join(str(x) for x in v))
        elif isinstance(v, dict):
            parts.append(json.dumps(v, ensure_ascii=False))
        else:
            parts.append(str(v))
    return "\n".join(p for p in parts if p)


def parse_json_file(path: Path, collection: str, visibility: str) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]

    chunks: list[Chunk] = []
    rel = path.name
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        text = _join_fields(
            item.get("title"),
            item.get("content"),
            item.get("evidence"),
            item.get("requirements"),
            item.get("responsibilities"),
            item.get("tags"),
        )
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                text=text,
                source_file=rel,
                chunk_id=_make_chunk_id(rel, i, text),
                category=item.get("category", collection),
                visibility=visibility,
                metadata_json=json.dumps(item, ensure_ascii=False)[:4000],
            )
        )
    return chunks


def parse_markdown_file(path: Path, collection: str, visibility: str) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    rel = path.name

    sections: list[tuple[str, str]] = []
    current_title = "top"
    current_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    chunks: list[Chunk] = []
    for i, (title, body) in enumerate(sections):
        if not body:
            continue
        text = f"## {title}\n{body}"
        chunks.append(
            Chunk(
                text=text,
                source_file=rel,
                chunk_id=_make_chunk_id(rel, i, text),
                category=collection,
                visibility=visibility,
                metadata_json=json.dumps({"section": title}, ensure_ascii=False),
            )
        )
    return chunks


def parse_directory(dir_path: Path, collection: str, visibility: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not dir_path.exists():
        logger.warning(f"目录不存在: {dir_path}")
        return chunks

    for path in sorted(dir_path.iterdir()):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            chunks.extend(parse_json_file(path, collection, visibility))
        elif suffix in {".md", ".markdown", ".txt"}:
            chunks.extend(parse_markdown_file(path, collection, visibility))
    return chunks