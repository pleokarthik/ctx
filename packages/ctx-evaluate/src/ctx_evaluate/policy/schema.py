from dataclasses import dataclass, fields, asdict


@dataclass
class InputQualityPolicy:
    min_chunk_relevance_score: float = 0.5
    min_top_chunk_score: float = 0.7
    max_duplicate_ratio: float = 0.2
    max_low_score_chunk_ratio: float = 0.3
    min_token_headroom: float = 0.15
    max_high_score_truncations: int = 0
    max_source_domains: int = 3
    llm_rewrite_risk_threshold: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InputQualityPolicy":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def default(cls) -> "InputQualityPolicy":
        return cls()
