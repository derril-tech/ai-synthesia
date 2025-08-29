# ARCH.md

## System Architecture — Synesthesia AI

### Overview
A multi-tenant, event-driven system that uses **CrewAI + LangGraph** for **cross-modal story generation**. Agents ensure narrative, visuals, and audio are aligned via iterative loops, with guardrails for brand fit, safety, and cost.

### Architecture Diagram
```
Next.js 14 (Mantine+Tailwind) ── REST/SSE/WS ─► FastAPI API Gateway
                                                │
                                                ├─► NATS Subjects
                                                │     (story.generate, prompt.optimize, media.image, media.audio,
                                                │      evaluate.coherence, safety.check, export.make)
                                                │
                                                └─► Workers (CrewAI+LangGraph):
                                                      • narrative-worker
                                                      • visual-worker
                                                      • audio-worker
                                                      • opt-worker (prompt optimizer)
                                                      • consistency-worker
                                                      • safety-worker
                                                      • export-worker
Data Plane:
  • Postgres 16 + pgvector (RLS by workspace_id)
  • OpenSearch (keyword + aggregations)
  • Redis (cache, rate limits, DLQ)
  • S3/R2 (raw + rendered assets)
Observability:
  • OpenTelemetry → Prometheus/Grafana; Sentry errors
Security:
  • KMS-encrypted brand kits, signed URLs, RLS, CSP/TLS/HSTS
```

### CrewAI + LangGraph Orchestration
- **Nodes:** Narrative, Visual, Audio, Optimizer, Consistency, Safety, Export.  
- **Edges:** confidence/threshold loops; retry/backoff; abort on safety fail.  
- **Shared Memory:** Postgres embeddings, brand kits, prior assets.  
- **Guardrails:** token/step budgets, brand enforcement, citation & safety checks.

### Data Model (Summary)
- **orgs/users/workspaces/memberships** for tenancy & roles.  
- **projects/brand_kits/prompts/runs** for content and configs.  
- **assets** (text/image/audio/video) with embeddings.  
- **evaluations/safety_flags** for QA.  
- **exports/audit_log** for governance.  
All tables RLS’d by workspace_id; embeddings via pgvector.

### Retrieval & Analysis
- Hybrid search (BM25+vector) over prior prompts/assets.  
- Consistency analyzers: text↔image↔audio semantic similarity, tone checks.  
- Gap detection for missing visuals/audio lines.  

### Generation
- **Narrative Worker:** script, captions.  
- **Visual Worker:** keyframes, upscaling, compositing.  
- **Audio Worker:** TTS, soundbeds, mixdown.  
- **Optimizer:** rewrites prompts until eval thresholds.  
- **Consistency Checker:** enforces alignment across modalities.  
- **Safety Worker:** NSFW, copyright, policy filters.  
- **Export Worker:** MP4/ZIP/JSON/PDF bundles.

### Frontend
- **Pages:** Dashboard, Projects, Brand Kits, Assets, Exports, Settings.  
- **Components:** Canvas (script+frames+audio), Optimizer, BrandKit, AssetGrid, ExportWizard.  
- **Realtime:** SSE for text; WS for job progress.  
- **Guardrails:** exports disabled until alignment & safety thresholds pass.

### APIs (selected)
- `POST /projects` — create new project.  
- `POST /storypacks/generate` — trigger generation; stream via SSE.  
- `POST /prompts/optimize` — improve prompts.  
- `POST /evaluations/score` — compute alignment & quality.  
- `POST /exports/bundle` — render MP4/ZIP/PDF/JSON.  
Standards: OpenAPI 3.1, Problem+JSON, cursor pagination, Idempotency-Key, SSE for long ops.

### Performance & Scaling
- Redis caches for brand kits and popular presets.  
- Pre-warm for “Ad kit,” “Lesson module.”  
- Adaptive early-exit on confidence.  
- Parallel TTS chunking; adaptive image steps.  
- SLO Targets: Story Pack < 45s p95; Export MP4 < 12s p95; success ≥ 99%.

### Security & Compliance
- RLS everywhere; tenant isolation.  
- KMS-encrypted secrets; signed URLs.  
- Safety filters, copyright detection.  
- GDPR/CCPA DSR endpoints; retention & erasure policies.

### Deployment
- FE: Vercel (ISR + Edge cache).  
- API/Workers: Render/Fly/GKE autoscaling pools.  
- CI/CD: GitHub Actions (lint/tests/build/scan/deploy); Terraform for infra.  
- Monitoring: OTel dashboards + Sentry; runbooks + paging.  
