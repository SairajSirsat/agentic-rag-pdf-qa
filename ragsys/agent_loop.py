from pydantic import BaseModel

import config
from ragsys.bm25_index import BM25Index
from ragsys.hybrid_retrieve import hybrid_retrieve
from ragsys.llm import chat_freeform, chat_structured
from ragsys.prompts import (
    ANSWER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JudgeResult,
    build_answer_user_msg,
    build_judge_user_msg,
)
from ragsys.rerank import rerank
from ragsys.vectorstore import VectorStore


class AnswerResult(BaseModel):
    answer: str
    thinking: str | None = None
    citations: list[int]
    iterations_used: int
    trace: list[dict]


def _print_iteration(entry: dict) -> None:
    print(f"\n--- Iteration {entry['iteration']} ---")
    print(f"Query: {entry['query']}")
    print(f"Dense/BM25 fused candidate ids: {entry['fused_ids']}")
    print("Reranked (id, score):")
    for cid, score in entry["reranked"]:
        print(f"  {cid}: {score:.4f}")
    print(f"Judge: {entry['judge']}")


def answer_question(question: str, pdf_id: str, verbose: bool = False) -> AnswerResult:
    vs = VectorStore(pdf_id)
    bm25 = BM25Index(pdf_id)
    if not bm25.load():
        raise RuntimeError(f"No BM25 index found for pdf_id={pdf_id}; ingest the PDF first.")

    accumulated: dict[str, dict] = {}
    query = question
    trace: list[dict] = []
    i = 1

    for i in range(1, config.MAX_ITERATIONS + 1):
        fused = hybrid_retrieve(query, vs, bm25)
        reranked = rerank(query, fused)

        for c in reranked:
            existing = accumulated.get(c["id"])
            if existing is None or c["rerank_score"] > existing["rerank_score"]:
                accumulated[c["id"]] = c

        context = sorted(accumulated.values(), key=lambda c: -c["rerank_score"])[: config.CONTEXT_TOP_K]

        judge = chat_structured(JUDGE_SYSTEM_PROMPT, build_judge_user_msg(question, context), JudgeResult)

        entry = {
            "iteration": i,
            "query": query,
            "fused_ids": [c["id"] for c in fused],
            "reranked": [(c["id"], c["rerank_score"]) for c in reranked],
            "judge": judge.model_dump(),
        }
        trace.append(entry)
        if verbose:
            _print_iteration(entry)

        if judge.sufficient or i == config.MAX_ITERATIONS:
            break
        query = judge.next_query or question

    reply = chat_freeform(ANSWER_SYSTEM_PROMPT, build_answer_user_msg(question, context))
    citations = sorted({c["metadata"]["page"] for c in context})

    return AnswerResult(
        answer=reply.content,
        thinking=reply.thinking,
        citations=citations,
        iterations_used=i,
        trace=trace,
    )
