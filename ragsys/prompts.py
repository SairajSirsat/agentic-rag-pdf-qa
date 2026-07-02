from pydantic import BaseModel


class JudgeResult(BaseModel):
    sufficient: bool
    reasoning: str
    next_query: str | None = None


JUDGE_SYSTEM_PROMPT = """You are assessing whether retrieved passages from a PDF are enough to answer a \
user's question. You will be given the question and a numbered list of context chunks, each tagged with \
its page number.

Decide:
- sufficient: true if the context contains enough information to fully and accurately answer the question.
- reasoning: one or two sentences on what is present or missing.
- next_query: if sufficient is false, propose a focused search query targeting the specific missing \
information (e.g. a named entity, section, or figure mentioned but not yet retrieved). If sufficient is \
true, leave this null.

Respond only with the requested structured output."""

ANSWER_SYSTEM_PROMPT = """You are answering a user's question using only the provided context chunks from a \
PDF, each tagged with its page number. Cite the page number for every claim inline as (p. N). If the \
context is incomplete or you are not fully confident, say so explicitly rather than guessing. Do not use \
outside knowledge beyond the provided context."""


def _format_context(context: list[dict]) -> str:
    lines = []
    for i, c in enumerate(context, start=1):
        page = c["metadata"].get("page", "?")
        lines.append(f"[{i}] (p. {page}) {c['text']}")
    return "\n\n".join(lines)


def build_judge_user_msg(question: str, context: list[dict]) -> str:
    return f"Question: {question}\n\nContext:\n{_format_context(context)}"


def build_answer_user_msg(question: str, context: list[dict]) -> str:
    return f"Question: {question}\n\nContext:\n{_format_context(context)}"
