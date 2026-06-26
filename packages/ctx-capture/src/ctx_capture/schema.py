import functools
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


def _flexible(cls):
    """Make dataclass __init__ accept and ignore unknown keyword arguments."""
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        valid = {f.name for f in fields(cls)}
        original_init(self, *args, **{k: v for k, v in kwargs.items() if k in valid})

    cls.__init__ = init
    return cls


@_flexible
@dataclass
class ChunkRecord:
    chunk_id: str
    source_doc_id: str
    content: str
    token_count: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_path: Optional[str] = None
    truncated: bool = False
    cache_hit: Optional[bool] = None


@_flexible
@dataclass
class TokenBudget:
    total_limit: int
    chunks_allocated: int
    history_allocated: int
    system_allocated: int
    headroom: int


@_flexible
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@_flexible
@dataclass
class Turn:
    role: str
    content: str
    tokens: Optional[int] = None


@_flexible
@dataclass
class CacheEvent:
    chunk_id: str
    hit: bool
    cache_source: Optional[str] = None


@_flexible
@dataclass
class RunRecord:
    query: str
    response: str
    chunks: Optional[list[ChunkRecord]] = None
    final_prompt: Optional[str] = None
    token_budget: Optional[TokenBudget] = None
    history_pre: Optional[list[Turn]] = None
    history_post: Optional[list[Turn]] = None
    eviction_reason: Optional[str] = None
    cache_events: Optional[list[CacheEvent]] = None
    model: Optional[str] = None
    token_usage: Optional[TokenUsage] = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "RunRecord":
        data = dict(data)
        if data.get("chunks") is not None:
            data["chunks"] = [ChunkRecord(**c) for c in data["chunks"]]
        if data.get("token_budget") is not None:
            data["token_budget"] = TokenBudget(**data["token_budget"])
        if data.get("history_pre") is not None:
            data["history_pre"] = [Turn(**t) for t in data["history_pre"]]
        if data.get("history_post") is not None:
            data["history_post"] = [Turn(**t) for t in data["history_post"]]
        if data.get("cache_events") is not None:
            data["cache_events"] = [CacheEvent(**e) for e in data["cache_events"]]
        if data.get("token_usage") is not None:
            data["token_usage"] = TokenUsage(**data["token_usage"])
        return cls(**data)
