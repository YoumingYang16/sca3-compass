"""FastAPI surface for the SCA3 Compass research and learning platform."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from threading import BoundedSemaphore
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .advanced_analytics import (
    bayesian_random_effects_meta,
    simulate_longitudinal_trial,
)
from .analytics import random_effects_meta, simulate_parallel_trial
from .audit_ledger import AuditLedger
from .design_assurance import AssuranceRequest, run_design_assurance
from .evidence_engine import evaluate_benchmark, query_evidence
from .experiment_store import ExperimentStore
from .fhir_service import (
    build_research_bundle,
    capability_statement,
    claim_to_evidence,
    provenance_for_resource,
    trial_to_research_study,
    validate_resource,
    validation_passed,
)
from .learning_engine import (
    initial_mastery,
    select_next_item,
    simulate_policy_comparison,
)
from .learning_sessions import LearningSessions, SessionConflict, public_item
from .literature_corpus import digest_json
from .molecular_api import router as molecular_router
from .repository import (
    PROJECT_ROOT,
    load_audit,
    load_claims,
    load_evidence_benchmark,
    load_evidence_evaluation,
    load_gene_expression_analysis,
    load_learning_evaluation,
    load_learning_model,
    load_learning_modules,
    load_progression_estimates,
    load_source_registry,
    load_transcriptomics_analysis,
    load_trials,
)
from .semantic_retrieval import REPORT_PATH, LiteratureIndex

AUDIT_LEDGER = AuditLedger(PROJECT_ROOT / "data" / "runtime" / "audit.sqlite3")
EXPERIMENTS = ExperimentStore(PROJECT_ROOT / "data" / "runtime" / "experiments.sqlite3")
LEARNING_SESSIONS = LearningSessions(PROJECT_ROOT / "data" / "runtime" / "learning.sqlite3")
DESIGN_SLOT = BoundedSemaphore(1)
RETRIEVAL_SLOT = BoundedSemaphore(1)

app = FastAPI(
    title="SCA3 Compass API",
    version="4.0.0",
    description=(
        "Public-data analytics, trial-design simulation, research informatics and "
        "adaptive learning for SCA3. Not a diagnostic or treatment service."
    ),
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(molecular_router)


class TrialSimulationRequest(BaseModel):
    participants_per_arm: int = Field(60, ge=5, le=1000)
    followup_months: int = Field(24, ge=6, le=60)
    treatment_reduction: float = Field(0.3, ge=0, le=1)
    slope_sd: float = Field(1.5, gt=0, le=10)
    measurement_sd: float = Field(0.7, ge=0, le=10)
    attrition_rate: float = Field(0.1, ge=0, lt=0.8)
    simulations: int = Field(1000, ge=100, le=20_000)
    seed: int = Field(202709, ge=0, le=2_147_483_647)


class LongitudinalSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    participants_per_arm: int = Field(80, ge=10, le=1000)
    followup_months: int = Field(24, ge=12, le=60)
    visit_interval_months: int = Field(6)
    treatment_reduction: float = Field(0.3, ge=0, le=1)
    baseline_mean: float = Field(15.0, ge=0, le=40)
    baseline_sd: float = Field(6.0, gt=0, le=20)
    slope_sd: float = Field(1.5, gt=0, le=10)
    measurement_sd: float = Field(0.8, gt=0, le=10)
    measurement_ar1: float = Field(0.45, ge=0, lt=0.95)
    baseline_slope_correlation: float = Field(0.25, gt=-0.9, lt=0.9)
    attrition_rate: float = Field(0.15, ge=0, lt=0.7)
    missingness: str = Field("MAR", pattern="^(MCAR|MAR)$")
    mar_strength: float = Field(0.35, ge=0, le=3)
    simulations: int = Field(1000, ge=100, le=20_000)
    seed: int = Field(202709, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def scheduled_visits(self) -> LongitudinalSimulationRequest:
        if self.visit_interval_months not in {3, 4, 6, 12}:
            raise ValueError("Visit interval must be 3, 4, 6 or 12 months")
        if self.followup_months % self.visit_interval_months or self.followup_months // self.visit_interval_months < 2:
            raise ValueError("Follow-up must contain at least 3 equally spaced scheduled visits")
        return self


class EvidenceQueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(3, ge=1, le=5)


class LearningResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=20, max_length=80)
    question_token: str = Field(min_length=20, max_length=80)
    selected_index: int = Field(ge=0, le=10)


class LearningSimulationRequest(BaseModel):
    learners: int = Field(1000, ge=100, le=100_000)
    steps: int = Field(12, ge=1, le=100)
    seed: int = Field(202709, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def computation_budget(self) -> LearningSimulationRequest:
        if self.learners * self.steps > 300_000:
            raise ValueError("Limit policy experiments to 300,000 learner-steps per request")
        return self


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "sca3-compass",
        "medical_device": False,
        "scientific_claim_tier": "PUBLIC_REAL_ONLY",
        "platform_version": "4.0.0-research",
        "audit_chain_valid": AUDIT_LEDGER.verify()["valid"],
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    trials = load_trials()
    exact = [row for row in trials if row["exact_sca3_text_match"]]
    audit = load_audit()
    return {
        "trial_candidates": len(trials),
        "exact_sca3_matches": len(exact),
        "records_with_results": sum(row["has_results"] for row in exact),
        "registered_public_sources": len(load_source_registry()["sources"]),
        "verified_public_sources": audit["summary"]["passed"],
        "regions": len(
            {
                country
                for row in exact
                for country in row["countries"].split(" | ")
                if country
            }
        ),
        "governance": {
            "research_claims": "PUBLIC_REAL only",
            "simulation": "generated output, public real parameters",
            "patient_workspace": "watermarked test fixtures only",
            "learning_effect": "requires a new consented human study",
        },
        "research_methods": {
            "frequentist_meta_analysis": "REML",
            "bayesian_meta_analysis": "normal-normal hierarchical quadrature",
            "trial_design": "posterior predictive design assurance + independent sequential null calibration",
            "learning_model": "Bayesian Knowledge Tracing",
            "evidence_intelligence": "citation-bound retrieval with abstention",
            "interoperability": "FHIR R5 independent structural validation + focused constraints",
        },
    }


@app.get("/api/trials")
def trials(
    exact_only: bool = True,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> dict[str, Any]:
    rows = load_trials()
    if exact_only:
        rows = [row for row in rows if row["exact_sca3_text_match"]]
    return {"total": len(rows), "items": rows[:limit], "provenance_tier": "PUBLIC_REAL"}


@app.get("/api/trials/summary")
def trial_summary() -> dict[str, Any]:
    rows = [row for row in load_trials() if row["exact_sca3_text_match"]]
    statuses = Counter(row["overall_status"] or "UNKNOWN" for row in rows)
    phases = Counter(
        phase
        for row in rows
        for phase in (
            row["phases"].split(" | ") if row["phases"] else ["NOT_APPLICABLE"]
        )
    )
    intervention_types = Counter(
        value
        for row in rows
        for value in (
            row["intervention_types"].split(" | ")
            if row["intervention_types"]
            else ["UNSPECIFIED"]
        )
    )
    return {
        "total": len(rows),
        "statuses": dict(statuses.most_common()),
        "phases": dict(phases.most_common()),
        "intervention_types": dict(intervention_types.most_common()),
        "provenance_tier": "PUBLIC_REAL",
    }


@app.get("/api/analytics/meta-analysis")
def meta_analysis() -> dict[str, Any]:
    result = random_effects_meta(load_progression_estimates(primary_only=True))
    result["outcome"] = "Annual SARA score change (points/year)"
    result["data_scope"] = "Independent primary publication-level estimates"
    result["provenance_tier"] = "PUBLIC_REAL_DERIVED"
    return result


@app.get("/api/analytics/bayesian-meta-analysis")
def bayesian_meta_analysis() -> dict[str, Any]:
    return bayesian_random_effects_meta(
        load_progression_estimates(primary_only=True), seed=202709
    )


@app.post("/api/analytics/trial-simulation")
def trial_simulation(request: TrialSimulationRequest) -> dict[str, Any]:
    meta = random_effects_meta(load_progression_estimates(primary_only=True))
    return simulate_parallel_trial(
        annual_slope=meta["pooled_effect"],
        participants_per_arm=request.participants_per_arm,
        followup_months=request.followup_months,
        treatment_reduction=request.treatment_reduction,
        slope_sd=request.slope_sd,
        measurement_sd=request.measurement_sd,
        attrition_rate=request.attrition_rate,
        simulations=request.simulations,
        seed=request.seed,
    )


@app.post("/api/analytics/longitudinal-trial-simulation")
def longitudinal_trial_simulation(
    request: LongitudinalSimulationRequest,
) -> dict[str, Any]:
    meta = random_effects_meta(load_progression_estimates(primary_only=True))
    result = simulate_longitudinal_trial(
        annual_slope=meta["pooled_effect"],
        **request.model_dump(),
    )
    AUDIT_LEDGER.append(
        actor="researcher",
        action="execute",
        resource_type="LongitudinalTrialSimulation",
        resource_id=f"seed-{request.seed}",
        metadata={
            "provenance_tier": result["provenance_tier"],
            "completed_simulations": result["completed_simulations"],
            "assumptions": result["assumptions"],
        },
    )
    return result


@app.get("/api/evidence")
def evidence() -> dict[str, Any]:
    return load_claims()


@app.post("/api/analytics/design-assurance")
def design_assurance(request: AssuranceRequest) -> dict[str, Any]:
    if not DESIGN_SLOT.acquire(blocking=False):
        raise HTTPException(429, "A design experiment is already running; retry after it completes")
    try:
        result = run_design_assurance(load_progression_estimates(primary_only=True), request)
        envelope = EXPERIMENTS.save(result)
        AUDIT_LEDGER.append(actor="local-researcher", action="execute", resource_type="DesignAssurance", resource_id=envelope["run_id"], metadata={"sha256": envelope["sha256"], "seed": request.seed})
        return envelope
    finally:
        DESIGN_SLOT.release()


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    return {"items": EXPERIMENTS.recent()}


@app.get("/api/experiments/{run_id}")
def experiment(run_id: str) -> dict[str, Any]:
    try:
        return EXPERIMENTS.get(run_id)
    except KeyError:
        raise HTTPException(404, "Experiment not found") from None
    except ValueError:
        raise HTTPException(409, "Experiment integrity check failed") from None


@app.post("/api/evidence/query")
def evidence_query(request: EvidenceQueryRequest) -> dict[str, Any]:
    result = query_evidence(request.query, top_k=request.top_k)
    AUDIT_LEDGER.append(
        actor="researcher",
        action="query",
        resource_type="EvidenceIndex",
        resource_id="bm25-v1",
        metadata={
            "action": result["action"],
            "citation_count": len(result["citations"]),
            "query_length": len(request.query),
        },
    )
    return result


@app.get("/api/evidence/evaluation")
def evidence_evaluation(refresh: bool = False) -> dict[str, Any]:
    if refresh:
        return evaluate_benchmark(load_evidence_benchmark())
    return load_evidence_evaluation()


@lru_cache(maxsize=1)
def literature_index() -> LiteratureIndex:
    return LiteratureIndex()


class LiteratureQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=600)
    method: Literal["bm25", "dense", "rrf", "diverse"] = "rrf"
    top_k: int = Field(default=6, ge=1, le=10)


@app.get("/api/literature")
def literature_summary() -> dict:
    try:
        return literature_index().summary()
    except FileNotFoundError:
        raise HTTPException(503, "Public literature corpus has not been built") from None
    except ValueError:
        raise HTTPException(409, "Literature index integrity check failed") from None


@app.post("/api/literature/search")
def literature_search(request: LiteratureQuery) -> dict:
    if not RETRIEVAL_SLOT.acquire(blocking=False):
        raise HTTPException(429, "A local literature search is running. Please retry shortly.")
    try:
        return literature_index().search(request.query, request.method, request.top_k)
    except FileNotFoundError:
        raise HTTPException(503, "Semantic files unavailable. Select BM25 or rebuild the local index.") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        RETRIEVAL_SLOT.release()


@app.get("/api/literature/passage/{passage_id}")
def literature_passage(passage_id: str) -> dict:
    index = literature_index()
    if passage_id not in index.id_to_index:
        raise HTTPException(404, "Passage not found")
    chunk = index.chunks[index.id_to_index[passage_id]]
    return {**chunk, "article": index.articles[chunk["pmcid"]]}


@app.get("/api/literature/benchmark")
def literature_benchmark() -> dict:
    if not REPORT_PATH.exists():
        return {"status": "not_run"}
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if digest_json({k: v for k, v in report.items() if k != "sha256"}) != report.get("sha256"):
        raise HTTPException(409, "Benchmark report integrity check failed")
    if report["corpus_sha256"] != literature_index().corpus["sha256"]:
        raise HTTPException(409, "Benchmark is stale for the current corpus")
    if not literature_index().manifest or report["index_sha256"] != literature_index().manifest["sha256"]:
        raise HTTPException(409, "Benchmark is stale for the current vector index")
    return {"status": "available", **report}


@app.get("/api/learning/modules")
def learning_modules() -> dict[str, Any]:
    return load_learning_modules()


@app.get("/api/learning/adaptive-model")
def adaptive_learning_model() -> dict[str, Any]:
    model = load_learning_model()
    mastery = initial_mastery(model)
    return {
        "schema_version": model["schema_version"],
        "model": model["model"],
        "evaluation_status": model["evaluation_status"],
        "concepts": model["concepts"],
        "initial_mastery": mastery,
        "next_item": public_item(select_next_item(model, mastery, [])),
        "item_count": len(model["items"]),
        "interpretation_boundary": (
            "BKT parameters are design assumptions until estimated with consented learner data."
        ),
    }


@app.post("/api/learning/respond")
def learning_response(request: LearningResponseRequest) -> dict[str, Any]:
    try:
        return LEARNING_SESSIONS.respond(request.session_id, load_learning_model(), request.question_token, request.selected_index)
    except KeyError:
        raise HTTPException(404, "Session not found or expired") from None
    except SessionConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/learning/sessions")
def create_learning_session() -> dict[str, Any]:
    try:
        return LEARNING_SESSIONS.start(load_learning_model())
    except SessionConflict as exc:
        raise HTTPException(429, str(exc)) from None


@app.get("/api/learning/sessions/{session_id}")
def get_learning_session(session_id: str) -> dict[str, Any]:
    try:
        return LEARNING_SESSIONS.get(session_id, load_learning_model())
    except KeyError:
        raise HTTPException(404, "Session not found or expired") from None
    except SessionConflict as exc:
        raise HTTPException(409, str(exc)) from None


@app.delete("/api/learning/sessions/{session_id}")
def delete_learning_session(session_id: str) -> dict[str, bool]:
    LEARNING_SESSIONS.delete(session_id)
    return {"deleted": True}


@app.post("/api/learning/policy-simulation")
def learning_policy_simulation(
    request: LearningSimulationRequest,
) -> dict[str, Any]:
    result = simulate_policy_comparison(load_learning_model(), **request.model_dump())
    AUDIT_LEDGER.append(
        actor="researcher",
        action="execute",
        resource_type="LearningPolicySimulation",
        resource_id=f"seed-{request.seed}",
        metadata={"learners": request.learners, "steps": request.steps},
    )
    return result


@app.get("/api/learning/policy-evaluation")
def learning_policy_evaluation() -> dict[str, Any]:
    return load_learning_evaluation()


@app.get("/api/genomics/datasets")
def genomics_datasets() -> dict[str, Any]:
    return {
        "provenance_tier": "PUBLIC_REAL",
        "items": [
            {
                "accession": "GSE309548",
                "organism": "Homo sapiens",
                "model": "post-mortem spinal cord",
                "files": 14,
                "groups": "6 SCA3 tissue-region files; 8 control tissue-region files",
                "analysis_boundary": "Repeated regions from a donor are not independent people.",
                "status": "donor-aware exploratory analysis complete",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE309548",
            },
            {
                "accession": "GSE93713",
                "organism": "Homo sapiens",
                "model": "patient-derived and genetically corrected iPSC",
                "files": 6,
                "groups": "3 SCA3 iPSC; 3 genetically corrected iPSC",
                "analysis_boundary": "Cell-model findings are not clinical predictions.",
                "status": "matrix endpoint verified; metadata ingested",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93713",
            },
        ],
    }


@app.get("/api/genomics/gse309548-analysis")
def gse309548_analysis() -> dict[str, Any]:
    return load_transcriptomics_analysis()


@app.get("/api/genomics/gse309548-gene-analysis")
def gse309548_gene_analysis() -> dict[str, Any]:
    return load_gene_expression_analysis()


@app.get("/api/fhir/research-bundle")
def fhir_research_bundle() -> dict[str, Any]:
    trials = [row for row in load_trials() if row["exact_sca3_text_match"]]
    return build_research_bundle(trials, load_claims()["claims"])


@app.get("/api/fhir/metadata")
def fhir_metadata() -> dict[str, Any]:
    return capability_statement()


@app.get("/api/fhir/ResearchStudy")
def fhir_research_studies(
    status: str | None = None,
    identifier: str | None = None,
    count: Annotated[int, Query(alias="_count", ge=1, le=100)] = 50,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> dict[str, Any]:
    rows = [row for row in load_trials() if row["exact_sca3_text_match"]]
    resources = [trial_to_research_study(row) for row in rows]
    if status:
        resources = [item for item in resources if item["status"] == status]
    if identifier:
        resources = [item for item in resources if any(identifier in {token["value"], token["system"] + "|" + token["value"]} for token in item["identifier"])]
    total = len(resources)
    resources = resources[:min(limit, count)]
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": total,
        **({"entry": [{"resource": item} for item in resources]} if resources else {}),
    }


@app.get("/api/fhir/Evidence")
def fhir_evidence() -> dict[str, Any]:
    resources = [claim_to_evidence(item) for item in load_claims()["claims"]]
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(resources),
        "entry": [{"resource": item} for item in resources],
    }


@app.post("/api/fhir/$validate")
def fhir_validate(resource: dict[str, Any]) -> dict[str, Any]:
    outcome = validate_resource(resource)
    AUDIT_LEDGER.append(
        actor="researcher",
        action="validate",
        resource_type=str(resource.get("resourceType", "Unknown")),
        resource_id=str(resource.get("id", "unidentified")),
        metadata={"valid": validation_passed(outcome), "validator": "fhir.resources 8.1.0 / R5 + local constraints"},
    )
    return outcome


@app.get("/api/fhir/validation-report")
def fhir_validation_report() -> dict[str, Any]:
    bundle = fhir_research_bundle()
    outcome = validate_resource(bundle)
    return {"valid": validation_passed(outcome), "counts": dict(Counter(entry["resource"]["resourceType"] for entry in bundle["entry"])), "outcome": outcome, "validator": "fhir.resources 8.1.0; R5 5.0.0 + local constraints"}


@app.get("/api/fhir/{resource_type}/{resource_id}")
def fhir_read(resource_type: str, resource_id: str) -> dict[str, Any]:
    studies = [trial_to_research_study(row) for row in load_trials() if row["exact_sca3_text_match"]]
    evidence = [claim_to_evidence(item) for item in load_claims()["claims"]]
    resources = studies + evidence
    if resource_type == "Provenance":
        resources = [provenance_for_resource(item) for item in resources]
    for resource in resources:
        if resource["resourceType"] == resource_type and resource["id"] == resource_id:
            return resource
    raise HTTPException(404, "Research resource not found")


@app.get("/api/governance/audit-ledger")
def audit_ledger(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> dict[str, Any]:
    verification = AUDIT_LEDGER.verify()
    return {
        "verification": verification,
        "events": AUDIT_LEDGER.events(limit),
        "storage": "local SQLite append-only ledger",
        "contains_patient_data": False,
    }


@app.get("/api/governance")
def governance() -> dict[str, Any]:
    registry = load_source_registry()
    audit = load_audit()
    return {
        "policy_version": "1.0",
        "sources": registry["sources"],
        "audit": audit["summary"],
        "claim_gate": [
            "registered provenance",
            "raw snapshot digest",
            "deterministic transformation",
            "schema and statistical tests",
            "effect size and uncertainty",
            "visible interpretation boundary",
        ],
        "prohibited": [
            "diagnosis",
            "individual disease-course prediction",
            "treatment recommendation",
            "fixture data in scientific inference",
            "simulation presented as observed patients",
        ],
    }


def run() -> None:
    uvicorn.run("sca3_compass.app:app", host="127.0.0.1", port=8000, reload=False)


WEB_DIST = PROJECT_ROOT / "apps" / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
