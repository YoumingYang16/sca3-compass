# Target enterprise architecture

## Product boundary

SCA3 Compass is a research, information-management and learning system. It is
not a medical device and does not provide diagnosis or treatment.

## Platform layers

```text
Public APIs and archives
  ClinicalTrials.gov | PMC | GEO | curated authoritative guidance
                            |
                            v
Ingestion and provenance control
  source registry | immutable raw zone | checksums | schema tests | lineage
                            |
          +-----------------+------------------+
          |                                    |
          v                                    v
Analytics workbench                     Evidence knowledge layer
  meta-analysis                           versioned claims
  transcriptomics + gene-level model       retrieval and citations
  uncertainty                             abstention and safety policy
  trial simulation                               |
          |                                      v
          +-----------------> API gateway and policy engine
                                      |
                +---------------------+---------------------+
                |                     |                     |
                v                     v                     v
        Research dashboard      Care workspace       Learning studio
        real-data results       FHIR-compatible      adaptive lessons
        reproducibility         provenance/audit     learning assessment
```

## Planned services

- `source-registry`: catalogues provenance, access and allowed uses.
- `ingestion-worker`: retrieves versioned public data and produces manifests.
- `analytics-api`: serves validated, immutable statistical result artifacts.
- `simulation-worker`: runs seeded trial simulations as queued jobs.
- `evidence-service`: resolves claims to source passages and enforces citation
  coverage and abstention.
- `fhir-gateway`: imports and exports supported FHIR resources and provenance.
- `learning-engine`: rule-based adaptation, mastery tracking and assessments.
- `web-app`: role-aware patient, caregiver, researcher and reviewer interfaces.
- `audit-service`: records access, transformations, model versions and reviews.

The first implementation can be a modular monolith with these boundaries. The
interfaces remain explicit so high-load workers can be separated later without
rewriting the domain model.

## Scientific work packages

### WP1: Registry and natural-history evidence

Ingest real public trial records and publication-level estimates. Build a
transparent extraction table with dual-entry review for critical numeric
fields.

### WP2: Statistical inference

Estimate comparable progression outcomes with random-effects meta-analysis,
heterogeneity, prediction intervals and leave-one-study-out sensitivity.

### WP3: Trial design laboratory

Use only estimates and uncertainty from WP2 or clearly cited public studies.
Run seeded longitudinal simulations across sample size, follow-up, attrition,
measurement schedule and treatment-effect assumptions.

### WP4: Human transcriptomics

Analyze GSE309548 with donor-aware methods and treat tissue regions from the
same donor as repeated observations. Use GSE93713 only as a separate cell-model
analysis. Cross-study findings are compared at pathway level rather than by
naively merging incompatible samples.

The implemented primary molecular artifact is a donor-pseudobulk gene model:
author-provided Kallisto estimated counts are mapped to GENCODE v50, summed by
donor, filtered by CPM, normalized as log2 CPM and fit with `SCA3 + sex`.
Residual variances receive across-gene empirical-Bayes moderation. An
age-adjusted sensitivity model, leave-one-donor-out direction checks and an
exact fixed-size label-assignment calibration are reported beside the result.
The eight-donor post-mortem cohort remains exploratory and cannot establish
causality or predict an individual.

### WP5: Information workflow

Represent public research evidence with interoperable resources. Patient-level
FHIR workflow testing uses TEST_FIXTURE records unless ethically obtained real
records become available; those fixtures never enter scientific outputs.

### WP6: Adaptive learning

Design from explicit learning objectives, formative user research, prototypes,
teach-back and delayed retention. The new product's learning effect requires a
NEW_HUMAN_STUDY dataset; public historical datasets can validate content but
cannot establish that users learned from this product.
