from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.app.domain.reports import StructuredReportSchema
from backend.app.research.types import ClassifiedEvidence, ScoredSource


@dataclass(frozen=True)
class SynthesisRequest:
    report_id: str
    company_id: str
    company_name: str
    normalized_company_name: str
    sources: list[ScoredSource]
    evidence: list[ClassifiedEvidence]
    include_cv: bool = False
    include_cv_tailoring: bool = False
    include_adapted_cv_draft: bool = False
    cv_text: str | None = None


class ReportSynthesizer(Protocol):
    def synthesize(self, request: SynthesisRequest) -> StructuredReportSchema:
        ...


class SynthesisError(RuntimeError):
    pass
