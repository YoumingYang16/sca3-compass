# Data provenance and claims policy

## Objective

Every number shown as a research result must be traceable to a public real-world
source, a transparent transformation, a versioned analysis, and a testable
claim. The platform must never blur real observations, newly collected human
responses, and software fixtures.

## Provenance tiers

### PUBLIC_REAL

Existing, publicly retrievable observations or aggregate results. Examples are
ClinicalTrials.gov records, PMC tables and GEO expression files. These data may
support scientific inference after limitations, licensing and study design are
reviewed.

### CONTROLLED_REAL

Real records that require an application, data-use agreement, credential, or
institutional approval. They are not required for the core project. Adding them
later must not silently change the reproducibility status of an analysis.

### NEW_HUMAN_STUDY

Responses created when people evaluate the new learning product. These data do
not exist in a public historical dataset. They require an appropriate consent
and ethics pathway and are isolated from clinical and omics datasets.

### SIMULATION_NOT_PATIENT_DATA

Mathematically generated observations with known synthetic truth, used for
operating-characteristic experiments and method evaluation. They may support
claims about behavior under the specified generator, never empirical claims
about patients or independent biological replication. See the molecular
methods protocol for the simulation-to-real-data boundary.

### TEST_FIXTURE

Invented records used solely for unit tests, FHIR conformance tests, security
tests and demonstrations. They are visibly watermarked and excluded by policy
from scientific inference.

## Claim gates

Before an analysis can be labelled reproducible, it must have:

1. a registered source and retrieval timestamp;
2. the original source URL and source-specific terms;
3. a raw-file digest or an API-response snapshot digest;
4. a deterministic transformation pipeline;
5. a data dictionary and exclusion log;
6. executable tests for schema and expected row counts;
7. a statistical analysis plan written before final analysis;
8. effect sizes and uncertainty, not only p-values;
9. an explicit statement of what the data cannot establish.

## Hard prohibitions

- TEST_FIXTURE records cannot enter meta-analysis, model training, power
  estimation, transcriptomic inference, or learning-effect claims.
- Aggregate publication results cannot be presented as patient-level data.
- Repeated tissue samples from one donor cannot be treated as independent
  people.
- Public cell-line or animal results cannot be presented as clinical effects in
  patients.
- A simulation can compare trial designs but cannot demonstrate that a therapy
  works.
- The evidence assistant cannot diagnose, predict an individual's course, or
  recommend treatment.

## Reproducibility output

Each published result will include a machine-readable manifest containing the
source identifiers, retrieval dates, checksums, code version, parameters,
random seed, environment lockfile and artifact checksums.
