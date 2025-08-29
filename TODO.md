# TODO.md

## Development Roadmap (5 Phases)

### Phase 1 — Foundations & Infrastructure ✅ COMPLETED
- [x] **Monorepo & Envs**: pnpm turborepo; apps: web/api/workers; dev/stage/prod.  
- [x] **Auth & Tenancy**: Email/SSO, orgs/workspaces, memberships; Casbin RBAC; RLS by workspace_id.  
- [x] **API Gateway**: FastAPI REST /v1, Problem+JSON, Idempotency-Key, ULID request IDs.  
- [x] **Data Stores**: Postgres 16 + pgvector; Redis (cache, DLQ); S3/R2; OpenSearch.  
- [x] **Frontend Setup**: Next.js 14, Mantine+Tailwind, TanStack Query, Zustand; dashboard skeleton.  
- [x] **Observability**: OTel spans; Prometheus/Grafana; Sentry.  
- [x] **CI/CD & IaC**: GitHub Actions (lint, typecheck, tests, Docker build/scan/deploy); Terraform for infra.  

### Phase 2 — Ingestion, Brand Kits & Normalization ✅ COMPLETED
- [x] **Connectors**: GDrive, Notion, Dropbox, S3/GCS, YouTube, RSS.  
- [x] **Normalization**: text cleaning, EXIF strip, audio normalization.  
- [x] **Brand Kits**: palette, typography, lexicon, SSML presets, logos; versioning with diffs.  
- [x] **Storage**: asset uploads via signed URLs; snapshots in Postgres.  
- [x] **Embeddings**: multimodal embeddings (CLIP/CLAP/LLM).  
- [x] **UI**: Brand Kit editor with previews.  

### Phase 3 — Multi-Agent Generation & Optimization ✅ COMPLETED
- [x] **Narrative Worker**: idea → outline → script → captions (streaming).  
- [x] **Visual Worker**: cover + keyframes via SDXL/DALL·E; upscaler.  
- [x] **Audio Worker**: SSML → TTS; soundbeds; mixdown loudness normalize.  
- [x] **Prompt Optimizer**: iterative rewriting w/ eval signals.  
- [x] **Consistency Worker**: cross-modal similarity, tone alignment, gap detection.  
- [x] **LangGraph Orchestration**: loop until thresholds met or budget exhausted.  
- [x] **UI**: Story Canvas (script · frames · audio) with inline "Fix Alignment."  

### Phase 4 — Evaluation, Safety & Exports ✅ COMPLETED
- [x] **Evaluations**: alignment scores; readability; audio MOS proxy; image quality metrics.  
- [x] **Safety Worker**: NSFW/toxicity/policy; copyright heuristics.  
- [x] **Exports**: MP4 story video; ZIP bundle; JSON metadata; PDF brief.  
- [x] **Reports**: Brand Fit, Cost, Alignment.  
- [x] **UI**: Export Wizard with template options.  
- [x] **Governance**: review states, comment threads, audit logs.  

### Phase 5 — Testing, Automation & Launch ✅ COMPLETED
- [x] **Automation**: presets, schedules, A/B variants; watches for drift/fails.  
- [x] **Testing**: unit (SSML, palette enforcement, audio mixdown), retrieval recall@k, rubric-based evals, integration (narrative→visual→audio→opt→export), E2E (Playwright).  
- [x] **Performance/Chaos**: concurrent runs, queue backpressure, provider outage, stale API keys.  
- [x] **Security**: signed URLs, per-workspace KMS keys, data deletion/export.  
- [x] **Launch**: staging hardening, canary deploy, runbooks, error budgets.  
