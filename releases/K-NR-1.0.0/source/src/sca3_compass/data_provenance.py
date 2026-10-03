"""Data provenance rules shared by ingestion, analytics, and product layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DataTier(StrEnum):
    """Mutually exclusive provenance tiers."""

    PUBLIC_REAL = "PUBLIC_REAL"
    CONTROLLED_REAL = "CONTROLLED_REAL"
    NEW_HUMAN_STUDY = "NEW_HUMAN_STUDY"
    TEST_FIXTURE = "TEST_FIXTURE"


class IntendedUse(StrEnum):
    SCIENTIFIC_INFERENCE = "SCIENTIFIC_INFERENCE"
    EVIDENCE_RETRIEVAL = "EVIDENCE_RETRIEVAL"
    LEARNING_EVALUATION = "LEARNING_EVALUATION"
    SOFTWARE_TESTING = "SOFTWARE_TESTING"
    UI_DEMONSTRATION = "UI_DEMONSTRATION"


ALLOWED_USES: dict[DataTier, frozenset[IntendedUse]] = {
    DataTier.PUBLIC_REAL: frozenset(
        {
            IntendedUse.SCIENTIFIC_INFERENCE,
            IntendedUse.EVIDENCE_RETRIEVAL,
            IntendedUse.SOFTWARE_TESTING,
            IntendedUse.UI_DEMONSTRATION,
        }
    ),
    DataTier.CONTROLLED_REAL: frozenset(
        {
            IntendedUse.SCIENTIFIC_INFERENCE,
            IntendedUse.SOFTWARE_TESTING,
        }
    ),
    DataTier.NEW_HUMAN_STUDY: frozenset(
        {
            IntendedUse.LEARNING_EVALUATION,
        }
    ),
    DataTier.TEST_FIXTURE: frozenset(
        {
            IntendedUse.SOFTWARE_TESTING,
            IntendedUse.UI_DEMONSTRATION,
        }
    ),
}


@dataclass(frozen=True)
class DataAsset:
    id: str
    tier: DataTier
    intended_uses: tuple[IntendedUse, ...]
    public_url: str | None
    license_or_terms: str

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> DataAsset:
        return cls(
            id=str(item["id"]),
            tier=DataTier(item["tier"]),
            intended_uses=tuple(IntendedUse(value) for value in item["intended_uses"]),
            public_url=item.get("url"),
            license_or_terms=str(item.get("license_or_terms", "unspecified")),
        )

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("A data asset must have a non-empty id")
        if self.tier is DataTier.PUBLIC_REAL and not self.public_url:
            raise ValueError(f"{self.id}: PUBLIC_REAL assets require a public URL")
        disallowed = set(self.intended_uses) - set(ALLOWED_USES[self.tier])
        if disallowed:
            labels = ", ".join(sorted(value.value for value in disallowed))
            raise ValueError(
                f"{self.id}: {self.tier.value} data cannot be used for {labels}"
            )


def validate_registry(items: list[dict[str, Any]]) -> list[DataAsset]:
    assets = [DataAsset.from_mapping(item) for item in items]
    ids = [asset.id for asset in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("Data source ids must be unique")
    for asset in assets:
        asset.validate()
    return assets
