import argparse
import os
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import ollama

import config
from ragsys.agent_loop import answer_question
from ragsys.ingest import ingest_pdf
from ragsys.registry import get_active, get_entry, list_pdfs, pdf_hash

_client = ollama.Client(host=config.OLLAMA_HOST)


def _check_ollama() -> None:
    try:
        _client.list()
    except Exception as e:
        print(f"Could not reach Ollama at {config.OLLAMA_HOST} — is `ollama serve` running?\n({e})")
        sys.exit(1)


def _resolve_pdf_path(pdf_arg: str) -> str:
    """Accept either a bare filename or a path already under pdfs/."""
    candidates = [pdf_arg, os.path.join(config.PDFS_DIR, pdf_arg)]
    for c in candidates:
        if os.path.isfile(c):
            return c
    print(f"Could not find PDF '{pdf_arg}' (looked in '.' and '{config.PDFS_DIR}/'). "
          f"Drop the PDF into '{config.PDFS_DIR}/' first.")
    sys.exit(1)


def cmd_setup(args: argparse.Namespace) -> None:
    _check_ollama()
    for model in (config.EMBED_MODEL, config.LLM_MODEL):
        print(f"Pulling {model} ...")
        for progress in _client.pull(model, stream=True):
            status = progress.get("status", "")
            print(f"  {status}")
    print("Setup complete. Run `ollama list` to confirm.")


def cmd_ingest(args: argparse.Namespace) -> None:
    _check_ollama()
    path = _resolve_pdf_path(args.pdf)
    ingest_pdf(path, force=args.force, verbose=True)


def cmd_list(args: argparse.Namespace) -> None:
    os.makedirs(config.PDFS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(config.PDFS_DIR) if f.lower().endswith(".pdf"))
    active = get_active()
    if not files:
        print(f"No PDFs found in '{config.PDFS_DIR}/'.")
        return
    for f in files:
        path = os.path.join(config.PDFS_DIR, f)
        pid = pdf_hash(path)
        entry = get_entry(pid)
        status = f"ingested, {entry['num_chunks']} chunks" if entry else "not ingested"
        marker = " (active)" if pid == active else ""
        print(f"{f}: {status}{marker}")


def _resolve_pdf_id(pdf_arg: str | None) -> str:
    if pdf_arg:
        path = _resolve_pdf_path(pdf_arg)
        pid = pdf_hash(path)
        if not get_entry(pid):
            print(f"'{pdf_arg}' hasn't been ingested yet; ingesting now...")
            return ingest_pdf(path, verbose=True)
        return pid
    active = get_active()
    if not active:
        print("No PDF has been ingested yet. Run `python main.py ingest <pdf_path>` first.")
        sys.exit(1)
    return active


def _print_answer(question: str, result) -> None:
    print(f"\nAnswer ({result.iterations_used} iteration(s), pages {result.citations}):\n{result.answer}\n")


def cmd_ask(args: argparse.Namespace) -> None:
    _check_ollama()
    pid = _resolve_pdf_id(args.pdf)

    if args.question:
        result = answer_question(args.question, pid, verbose=args.verbose)
        _print_answer(args.question, result)
        return

    print("Interactive mode — type a question, or 'exit'/'quit' to stop.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        result = answer_question(question, pid, verbose=args.verbose)
        _print_answer(question, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG Q&A over a single PDF.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Pull required Ollama models.").set_defaults(func=cmd_setup)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF from the pdfs/ folder.")
    p_ingest.add_argument("pdf", help="Path or filename under pdfs/")
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if already indexed.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask questions about an ingested PDF.")
    p_ask.add_argument("--pdf", default=None, help="Path or filename under pdfs/; defaults to the active PDF.")
    p_ask.add_argument("--question", default=None, help="Ask a single question and exit.")
    p_ask.add_argument("--verbose", action="store_true", help="Print retrieval/rerank/judge trace.")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("list", help="List PDFs in pdfs/ and their ingestion status.").set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
