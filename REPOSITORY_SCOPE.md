# Curated source/evidence snapshot

Initial private repository; not a completed R5 method release. No open-source license selected.
Includes local product source, selected research source/proofs, completed freeze manifests,
summaries and derived D020 scalar records in apps/web/public/research-evidence.json.
V1/R2/R3/R4 frozen bytes copied unchanged where included. R5 stays DEVELOPMENT.

NOT INCLUDED: original observation arrays, full raw experiments, biological datasets,
future/held-out data, environments, local runtime databases, old terminal logs and model weights.
The original complete archives remain on the owner's machine. Historical full replay commands
may require these omitted artifacts and original Windows-path provenance. This snapshot cannot
rerun all confirmations or re-export the display JSON without those archives; do not claim it can.
The static portal builds directly from the included exported JSON. No data ingestion is necessary.

Build: cd apps/web; pnpm install --frozen-lockfile; pnpm build:public.
Serve only dist-public; never publish FastAPI as-is.
GitHub upload does not deploy a public website. PUBLIC_DEPLOYMENT_SECURITY.md gives the boundary.
Manifest hashes identify this export, not correctness, novelty, or clinical benefit.
