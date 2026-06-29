from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class MetricContribution:
    metric: str
    raw_value: float | None
    normalized_value: float | None
    direction: str
    weight: float
    contribution: float
    contribution_pct: float | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DimensionScore:
    dimension_id: str
    score: float
    percentile: float | None
    weight: float
    metric_contributions: list[MetricContribution]
    confidence: float | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreConfidence:
    confidence_index: float
    bootstrap_stability: float | None
    weight_stability: float | None
    minutes_reliability: float | None
    population_reliability: float | None
    metric_coverage: float | None
    data_quality: float | None
    validation_status: str
    confidence_interval: dict[str, float | None]
    quality_label: str


@dataclass(frozen=True)
class ScoreExplanation:
    positive_contributors: list[MetricContribution]
    negative_contributors: list[MetricContribution]
    dimension_contributions: list[dict[str, Any]]
    strengths: list[str]
    weaknesses: list[str]
    review_flags: list[str]


@dataclass(frozen=True)
class RoleCalibration:
    role: str
    percentile_scope: str
    score_version: str
    calibration_status: str
    benchmark_population: dict[str, Any]


@dataclass(frozen=True)
class PlayerScore:
    player_id: str
    player_name: str | None
    role: str
    score_name: str
    raw_score: float
    normalized_score: float
    percentile: float
    dimension_scores: list[DimensionScore]
    confidence: ScoreConfidence
    explanation: ScoreExplanation
    calibration: RoleCalibration
    quality_flags: list[str]
    metadata: dict[str, Any]

    def to_api_object(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "role": self.role,
            "score_name": self.score_name,
            "scores": {
                "raw": self.raw_score,
                "normalized": self.normalized_score,
                "percentile": self.percentile,
            },
            "confidence": self.confidence.__dict__,
            "dimensions": [d.__dict__ for d in self.dimension_scores],
            "explanation": self.explanation.__dict__,
            "calibration": self.calibration.__dict__,
            "quality_flags": self.quality_flags,
            "metadata": self.metadata,
        }


class ScoreEngine:
    """Configuration-driven production score-engine skeleton.

    This class intentionally contains no hardcoded football metrics or coefficients.
    Runtime callers must provide score definitions generated from research artefacts.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def score_definitions(self) -> Mapping[str, Any]:
        return self.config.get("score_definitions", {})

    def validate_config(self) -> list[str]:
        errors: list[str] = []
        required = ["engine_version", "score_definitions", "confidence_framework", "percentile_rules", "eligibility_rules"]
        for key in required:
            if key not in self.config:
                errors.append(f"missing_config_key:{key}")
        for role, definition in self.score_definitions().items():
            if "dimensions" not in definition:
                errors.append(f"missing_dimensions:{role}")
            if "metrics" not in definition:
                errors.append(f"missing_metrics:{role}")
        return errors

    def explain_contract(self) -> dict[str, Any]:
        return {
            "contract": "ScoreEngine returns PlayerScore.to_api_object() payloads",
            "config_driven": True,
            "hardcoded_metrics": False,
            "required_inputs": [
                "player identity",
                "role",
                "raw metrics",
                "normalization config",
                "metric directions",
                "metric weights",
                "dimension weights",
                "confidence inputs",
                "calibration population",
            ],
        }
