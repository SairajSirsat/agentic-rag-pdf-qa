from typing import NamedTuple

import ollama
from pydantic import BaseModel

import config

_client = ollama.Client(host=config.OLLAMA_HOST)


class FreeformReply(NamedTuple):
    content: str
    thinking: str | None


def chat_structured(system: str, user: str, schema: type[BaseModel], model: str = config.LLM_MODEL) -> BaseModel:
    response = _client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema.model_json_schema(),
        think=config.JUDGE_THINK,
        options={"temperature": 0},
    )
    return schema.model_validate_json(response.message.content)


def chat_freeform(system: str, user: str, think: bool = config.ANSWER_THINK, model: str = config.LLM_MODEL) -> FreeformReply:
    response = _client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        think=think,
    )
    return FreeformReply(content=response.message.content, thinking=response.message.thinking)
