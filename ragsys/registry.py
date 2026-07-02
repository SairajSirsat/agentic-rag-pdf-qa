import hashlib
import json
import os
import time

import config


def pdf_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ensure_data_dir() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)


def load_registry() -> dict:
    _ensure_data_dir()
    if not os.path.exists(config.REGISTRY_PATH):
        return {"pdfs": {}, "active": None}
    with open(config.REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(reg: dict) -> None:
    _ensure_data_dir()
    with open(config.REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def register_pdf(pdf_id: str, source_path: str, num_chunks: int) -> None:
    reg = load_registry()
    reg["pdfs"][pdf_id] = {
        "source_path": source_path,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_chunks": num_chunks,
    }
    save_registry(reg)


def is_ingested(pdf_id: str) -> bool:
    reg = load_registry()
    return pdf_id in reg["pdfs"]


def set_active(pdf_id: str) -> None:
    reg = load_registry()
    reg["active"] = pdf_id
    save_registry(reg)


def get_active() -> str | None:
    reg = load_registry()
    return reg.get("active")


def get_entry(pdf_id: str) -> dict | None:
    reg = load_registry()
    return reg["pdfs"].get(pdf_id)


def list_pdfs() -> dict:
    return load_registry()["pdfs"]
