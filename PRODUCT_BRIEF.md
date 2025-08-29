 Synesthesia AI.  

1) Product Description & Presentation 

One-liner 

“Generate perfectly aligned stories across image, text, and audio—continuously optimized by prompt-smart agents.” 

What it produces 

Story Packs: narrative (Markdown), cover/scene images, narration & background audio. 

Campaign Kits: ad copy variations, hero/thumbnail images, 15–30s audio stingers. 

Lesson Modules: explainer script, diagrams/illustrations, narrated audio (SSML), quiz items. 

Brand Sets: copy in a defined tone/voice, style-consistent images, branded sonic logo. 

Export Bundles: MP4 story videos, ZIPs of assets + JSON metadata, shareable pages. 

Scope/Safety 

Decision support & content generation. Human review recommended for regulated domains. 

Alignment & safety: toxicity/policy filters, copyright/style-risk checks, source/asset tracking. 

Data boundaries: brand kits & prompts are tenant-scoped; read-only connectors by default. 

 

2) Target User 

Creators & Agencies: YouTubers, podcasters, indie authors, social teams. 

Marketing & Growth: landing pages, ads, product launches needing multi-asset consistency. 

Education & L&D: teachers, course builders, internal enablement teams. 

SMB SaaS: founders producing release notes, thumbnails, voiceovers in one pass. 

 

3) Features & Functionalities (Extensive) 

Ingestion & Connectors 

Inputs: single idea prompt; outline/script; sample assets (reference image/audio); brand kit (palette, logo, voice/tone guidelines); CSV of product facts. 

Connectors: Google Drive/Docs, Notion, Dropbox, S3/GCS, YouTube (caption/audio ingest), RSS, public URLs. 

Normalization: text cleaning & language detection; image EXIF strip; audio loudness & sample-rate normalize; format unification (WAV 16-bit 24kHz; PNG/JPEG). 

Versioning: snapshot tables for brand kits, prompts, assets; diff views for prompt changes. 

Enrichment 

Taxonomies & NER: domain, persona, sentiment, style tags (e.g., “wholesome,” “noir”). 

Style Embeddings: compute multi-modal embeddings for reference image/audio/text (CLAP/CLIP + LLM). 

Brand Rules: palette + typography + lexicon; SSML prosody presets; logo placement masks. 

Retrieval & Analysis 

Hybrid search: BM25 + vector for prior assets & prompts; cross-encoder re-rank for best matches. 

Consistency analyzers: text↔image↔audio semantic similarity; tone/reading-level checks; voiceprint similarity. 

Gap detection: find scenes missing visuals or lines without matching audio cues. 

Generation & Alignment 

Narrative Agent: outline → script → captions (streaming). 

Visual Agent: cover + keyframes (SDXL/DALL·E adapter), optional upscaling. 

Audio Agent: TTS narration (SSML), soundbeds/SFX (music model adapter), loudness mix. 

Prompt Optimizer Agent: iterative rewriting w/ eval signals (coherence, brand fit, cost/latency). 

Consistency Loop: LangGraph loops until thresholds met or budget cap reached. 

Validation & Scoring 

Alignment score: cross-modal similarity, brand/tone adherence, reading-level fit. 

Safety: NSFW/toxicity/policy; copyright heuristics (logo detection, lyric overlap). 

Quality gates: min MOS proxy for audio, min image quality (NIQE/BRISQUE proxy), factual flags. 

Views & Reporting 

Story Canvas: side-by-side script, frames, waveform; redlines for misalignments. 

Brand Fit Report: terminology adherence, palette usage, voice tone deltas. 

Cost Report: tokens, image steps, audio seconds; per-modal breakdown. 

Exports: MP4 story, ZIP assets, PDF brief, JSON bundle. 

Rules & Automations 

Presets: “Podcast intro,” “Lesson module,” “Ad kit.” 

Schedules: weekly content cadences; auto-refresh variations; A/B variants. 

Watches: notify on model/pipeline drift or failed quality gates. 

Collaboration & Governance 

Workspaces & roles: Owner/Admin/Editor/Viewer; share links with expiry. 

Review states: draft → in review → approved → published; comment threads; audit log. 

Human-in-the-loop: approve/override prompts; lock sections; rollback versions. 

 

4) Backend Architecture (Extremely Detailed & Deployment-Ready) 

4.0 Multi-Agent Framework & Orchestration 

CrewAI for role-based agents & tool use; LangGraph for DAG/state, retries, and optimization loops. 

Agents as nodes: Narrative, Visual, Audio, Prompt Optimizer, Consistency Checker, Safety, Exporter. 

Shared memory: prior assets/prompts, brand kits, evaluations in Postgres/pgvector; optional LanceDB for local vector dev. 

Budgets/guardrails: per-run token/step caps; allow-listed model/tools per agent; “facts & constraints first” step before freeform generation.  

4.1 Topology 

API Gateway: FastAPI (Python 3.11), REST /v1, Problem+JSON errors, RBAC (Casbin), Idempotency-Key, Request-ID (ULID). 

Workers (Python + CrewAI/LangGraph): 

narrative-worker (script/outlines, streaming) 

visual-worker (image/gen/upscale/composite) 

audio-worker (TTS/music mixdown) 

opt-worker (prompt optimization, eval) 

consistency-worker (cross-modal scoring) 

safety-worker (policy/content checks) 

export-worker (MP4/PDF/ZIP/JSON) 

Event bus/queues: NATS subjects: story.generate, prompt.optimize, media.image, media.audio, evaluate.coherence, safety.check, export.make; DLQ via Redis Streams. 

Datastores: 

Postgres 16 + pgvector (RLS by workspace_id) 

OpenSearch for keyword + aggregations 

S3/R2 for raw & rendered assets 

Redis for cache/session/rate limits 

ClickHouse (optional) for usage analytics 

Observability: OpenTelemetry → Prometheus/Grafana; Sentry for errors. 

Secrets: Cloud KMS; per-provider API keys; envelope encryption for brand kits. 

4.2 Data Model (Postgres + pgvector) 

-- Tenancy 
CREATE TABLE orgs( 
  id UUID PRIMARY KEY, name TEXT NOT NULL, plan TEXT DEFAULT 'free', 
  created_at TIMESTAMPTZ DEFAULT now() 
); 
CREATE TABLE users( 
  id UUID PRIMARY KEY, org_id UUID REFERENCES orgs(id) ON DELETE CASCADE, 
  email CITEXT UNIQUE NOT NULL, name TEXT, role TEXT DEFAULT 'member', 
  tz TEXT, created_at TIMESTAMPTZ DEFAULT now() 
); 
CREATE TABLE workspaces( 
  id UUID PRIMARY KEY, org_id UUID REFERENCES orgs(id) ON DELETE CASCADE, 
  name TEXT, created_by UUID REFERENCES users(id), 
  created_at TIMESTAMPTZ DEFAULT now() 
); 
CREATE TABLE memberships( 
  user_id UUID REFERENCES users(id), 
  workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE, 
  role TEXT CHECK (role IN('owner','admin','editor','viewer')), 
  PRIMARY KEY(user_id, workspace_id) 
); 
 
-- Projects & brand kits 
CREATE TABLE projects( 
  id UUID PRIMARY KEY, workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE, 
  name TEXT, type TEXT, preset TEXT, state TEXT DEFAULT 'draft', 
  created_by UUID REFERENCES users(id), created_at TIMESTAMPTZ DEFAULT now() 
); 
CREATE TABLE brand_kits( 
  id UUID PRIMARY KEY, workspace_id UUID, name TEXT, 
  palette JSONB, typography JSONB, lexicon JSONB, voice JSONB, logo_s3 TEXT, 
  created_at TIMESTAMPTZ DEFAULT now() 
); 
 
-- Prompts & runs 
CREATE TABLE prompts( 
  id UUID PRIMARY KEY, workspace_id UUID, project_id UUID REFERENCES projects(id) ON DELETE CASCADE, 
  purpose TEXT, text TEXT, meta JSONB, embedding VECTOR(1536), 
  created_at TIMESTAMPTZ DEFAULT now() 
); 
CREATE INDEX ON prompts USING ivfflat(embedding vector_cosine_ops); 
 
CREATE TABLE runs( 
  id UUID PRIMARY KEY, project_id UUID, started_at TIMESTAMPTZ DEFAULT now(), 
  status TEXT, budget_tokens INT, budget_steps INT, costs JSONB, meta JSONB 
); 
CREATE TABLE run_steps( 
  id UUID PRIMARY KEY, run_id UUID REFERENCES runs(id) ON DELETE CASCADE, 
  agent TEXT, input JSONB, output JSONB, score NUMERIC, created_at TIMESTAMPTZ DEFAULT now() 
); 
 
-- Assets 
CREATE TABLE assets( 
  id UUID PRIMARY KEY, project_id UUID, kind TEXT CHECK(kind IN('text','image','audio','video')), 
  s3_path TEXT, thumb_s3 TEXT, format TEXT, duration_ms INT, width INT, height INT, 
  meta JSONB, embedding VECTOR(1536), created_at TIMESTAMPTZ DEFAULT now() 
); 
 
-- Evaluations & safety 
CREATE TABLE evaluations( 
  id UUID PRIMARY KEY, project_id UUID, run_id UUID, alignment NUMERIC, brand_fit NUMERIC, 
  readability NUMERIC, audio_mos_proxy NUMERIC, image_quality NUMERIC, notes TEXT, meta JSONB 
); 
CREATE TABLE safety_flags( 
  id UUID PRIMARY KEY, project_id UUID, run_id UUID, type TEXT, severity TEXT, details JSONB, created_at TIMESTAMPTZ DEFAULT now() 
); 
 
-- Exports & audit 
CREATE TABLE exports( 
  id UUID PRIMARY KEY, project_id UUID, type TEXT, s3_path TEXT, meta JSONB, created_at TIMESTAMPTZ DEFAULT now() 
); 
CREATE TABLE audit_log( 
  id BIGSERIAL PRIMARY KEY, org_id UUID, user_id UUID, action TEXT, target TEXT, meta JSONB, 
  created_at TIMESTAMPTZ DEFAULT now() 
); 
  

4.3 API Surface (REST /v1, OpenAPI 3.1) 

Auth/Users 

POST /auth/login · POST /auth/refresh · GET /me · GET /usage 

Projects & Inputs 

POST /projects {name, preset?, brand_kit_id?} 

POST /projects/{id}/inputs {prompt|outline|assets} (signed upload URLs) 

Generation 

POST /storypacks/generate {project_id, constraints?, refs?} → {run_id, pack_id} (SSE stream) 

GET /storypacks/{id} 

Optimization & Evaluations 

POST /prompts/optimize {project_id, objective} → {prompt_id, diff} 

POST /evaluations/score {project_id, asset_ids[]} → scores 

Exports 

POST /exports/bundle {project_id, type:"zip|json|mp4|pdf"} → signed URL 

Automation 

POST /schedules {project_id, cadence, preset} 

GET /runs?project_id=... 

Conventions: cursor pagination, Idempotency-Key, Problem+JSON, SSE for long jobs. 

4.4 Pipelines & Workers 

Narrative: idea → outline → script → captions (stream). 

Visual: prompt → image(s) → upscaler → compositing with logo/brand masks. 

Audio: SSML → TTS → music bed → mixdown → loudness normalize (EBU R128). 

Optimize: collect evals → rewrite prompts → re-run targeted stages. 

Consistency: compute cross-modal similarity; request fixes if below threshold. 

Safety: policy checks, copyright/brand conflicts; redline report. 

Export: render MP4 story (scenes + narration), build ZIP/PDF/JSON bundle. 

4.5 Realtime 

WebSockets: ws:workspace:{id}:run:{run_id} for pipeline progress. 

SSE: streaming text generation & optimization deltas; progress for media stages. 

4.6 Caching & Performance 

Redis caches for brand kits, prompt presets, and last-good prompts. 

Pre-warm popular presets (“ad kit,” “lesson”). 

Budget manager: early-exit on high confidence; adaptive image steps; audio chunking with parallel TTS. 

4.7 Observability 

OTel spans: narrative.gen, visual.gen, audio.tts, opt.loop, consistency.score, export.render. 

KPIs: alignment score p95, export latency p95, cost per pack, retry rates. 

4.8 Security & Compliance 

TLS/HSTS/CSP; signed URLs for assets; KMS-encrypted brand kits; RLS on workspace_id. 

Content safety profiles; configurable retention; right-to-erasure endpoint. 

 

5) Frontend Architecture (React 18 + Next.js 14) 

5.1 Tech Choices 

UI: Mantine (primary), Tailwind utilities where convenient. 

State: TanStack Query (server), Zustand (UI). 

Media: ffmpeg.wasm (client previews), wavesurfer.js (audio), react-player for quick MP4 preview. 

Realtime: SSE for text, WS for job progress. 

Forms/Validation: Zod + React Hook Form. 

5.2 App Structure 

/app 
  /(marketing)/page.tsx 
  /(auth)/sign-in/page.tsx 
  /(app)/dashboard/page.tsx 
  /(app)/projects/[id]/canvas/page.tsx 
  /(app)/brand-kits/page.tsx 
  /(app)/assets/page.tsx 
  /(app)/exports/page.tsx 
  /(app)/settings/page.tsx 
/components 
  Canvas/*           # script + frames + audio timeline 
  Optimizer/*        # prompt diffs, scores, suggestions 
  BrandKit/*         # palette, voice, lexicon editors 
  AssetGrid/*        # filterable library 
  ExportWizard/*     # MP4/ZIP/PDF builder 
/lib 
  api-client.ts 
  sse-client.ts 
  zod-schemas.ts 
  rbac.ts 
/store 
  useRunStore.ts 
  useProjectStore.ts 
  useBrandKitStore.ts 
  

5.3 Key Pages & UX Flows 

Dashboard: recent runs, cost/usage, “Create Story Pack” CTA. 

Canvas: 3-pane editor (Script · Frames · Audio). Inline “Fix Alignment” actions. 

Brand Kits: tone/voice (with live SSML preview), palette & logo upload, forbidden words. 

Assets: search by tags/modality; drag-into-canvas; version history. 

Exports: choose template → render → share link. 

5.4 Component Breakdown (Selected) 

Canvas/ScriptPanel.tsx — streaming text, readability score, edit/approve. 

Canvas/FrameStrip.tsx — keyframes, upscale, regenerate region. 

Canvas/AudioTrack.tsx — waveform scrub, re-synthesize selection, ducking slider. 

Optimizer/DiffCard.tsx — prompt revisions with before/after & score deltas. 

ExportWizard/Modal.tsx — template → background render → signed URL. 

5.5 Data Fetching & Caching 

Server Components for heavy list pages; client queries for live canvas. 

Prefetch: project → brand kit → last run → assets → scores. 

SWR for asset thumbs; CDN caching on images/audio snippets. 

5.6 Validation & Error Handling 

Zod schemas for project inputs; inline Problem+JSON renderer with remediation steps. 

Guardrails: export disabled until alignment & safety thresholds met. 

5.7 Accessibility & i18n 

Keyboard nav across panels; caption support; high-contrast theme; localized dates/units. 

 

6) SDKs & Integration Contracts 

Generate Story Pack 

 POST /v1/storypacks/generate 

{ 
  "project_id": "a2f9-...", 
  "constraints": { 
    "readingLevel": "grade8", 
    "imageStyle": "watercolor", 
    "voice": "warm_female", 
    "durationSec": 60 
  }, 
  "refs": {"brand_kit_id": "bk_123", "asset_ids": ["img1","aud1"]} 
} 
  

Response 

{"run_id":"01HY...","pack_id":"sp_9fd","status":"running"} 
  

Optimize Prompt 

 POST /v1/prompts/optimize → returns suggested prompt + rationale and expected score delta. 

Fetch Assets 

 GET /v1/assets?project_id=...&kind=image|audio|text 

Export 

 POST /v1/exports/bundle { "project_id": "sp_9fd", "type": "mp4" } → { "url": "https://signed..." } 

 

7) DevOps & Deployment 

FE: Vercel (ISR for marketing; Edge cache). 

APIs/Workers: Render/Fly.io/GKE per worker pool; autoscale by queue depth. 

DB: Managed Postgres + pgvector; PITR; read replicas. 

Search: Managed OpenSearch; daily snapshots. 

Cache/Bus: Redis + NATS; DLQ with backoff/jitter. 

Storage/CDN: S3/R2 + CloudFront/Cloudflare. 

CI/CD: GitHub Actions (lint, typecheck, unit/integration, Docker build/scan/sign, deploy); blue/green; alembic migrations approvals. 

IaC: Terraform modules for DB/Search/Redis/NATS/buckets/secrets/DNS. 

Envs: dev/staging/prod; feature flags; error budgets & paging. 

Operational SLOs 

Story Pack p95 (script+2 images+60s audio): < 45s 

Export MP4 p95: < 12s 

Search API p95: < 1.2s 

Pipeline success: ≥ 99% monthly 

 

8) Testing 

Unit: SSML generation rules; palette/lexicon enforcement; audio mixdown loudness. 

Retrieval: recall@k for prompt/asset fetch; ablations (keyword vs hybrid + re-rank). 

Generation: rubric-based human eval (coherence, tone, usefulness); inter-rater reliability. 

Cross-Modal: cosine similarity between modalities; threshold calibration. 

Integration: narrative → visual → audio → optimize → export. 

E2E (Playwright): create project → upload brand kit → generate → fix alignment → export MP4. 

Load: concurrent runs; queue backpressure; spillover to DLQ. 

Chaos: provider outage; stale API keys; model timeout; retry/backoff behavior. 

Security: RLS coverage; signed URL scope; audit completeness. 

 

9) Success Criteria 

Product KPIs 

≥ 85% of Story Packs pass alignment & safety on first try. 

Time-to-Publish: < 10 min from idea to exported MP4 (median). 

Creator satisfaction (CSAT) ≥ 4.5/5; Repeat use ≥ 60% weekly. 

Cost per Pack reduced ≥ 25% after optimizer warm-up. 

Engineering SLOs 

Error rate < 0.8% of runs; p95 latencies within targets. 

Data freshness for brand kits/prompts immediate; assets CDN hit-rate ≥ 80%. 

 

10) Visual/Logical Flows 

A) Input & Preparation 

 Prompt/outline + brand kit + refs → normalize → compute style embeddings → persist. 

B) Multi-Agent Generation (CrewAI + LangGraph) 

 Narrative Agent streams script → Visual Agent creates cover/frames → Audio Agent synthesizes narration/music → Consistency Checker scores cross-modal alignment → Prompt Optimizer rewrites prompts and re-invokes targeted stages until thresholds/budget. 

C) Safety & Review 

 Safety Worker flags issues → UI redlines → human approves/edits → lock sections. 

D) Export & Automations 

 Exporter renders MP4/ZIP/PDF/JSON → signed URL → schedule next pack or A/B variants → usage & cost telemetry logged. 

 

 