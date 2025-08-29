# PLAN.md

## Product: Synesthesia AI — Aligned Story Generation Across Modalities

### Vision & Goals
Create a **multi-agent system** (CrewAI + LangGraph) that generates **perfectly aligned multi-modal stories** (text, images, audio), optimized iteratively with prompt-smart agents, while enforcing **brand, safety, and consistency guardrails**.

### Key Objectives
- **Story Packs:** narrative text + cover/scene images + narration & background audio.
- **Campaign Kits:** ad copy variations, hero/thumbnail images, audio stingers.
- **Lesson Modules:** explainer script, diagrams, narrated audio, quiz items.
- **Brand Sets:** tone-consistent copy, style-consistent visuals, branded sonic logos.
- **Exports:** MP4 videos, ZIP bundles, JSON metadata, shareable links.
- **Governance:** human-in-the-loop reviews, audit trails, alignment & safety scoring.

### Target Users
- Creators & Agencies (YouTubers, podcasters, authors, social teams).
- Marketing & Growth teams (ads, launches, landing pages).
- Education & L&D (course builders, teachers).
- SMB SaaS (release notes, thumbnails, audio intros).

### High-Level Approach
1. **Frontend (Next.js 14 + React 18)**  
   - Project dashboard, Story Canvas (script + frames + audio timeline), Brand Kit editor.  
   - Realtime generation streams (SSE/WS), inline redlines for misalignments.  
   - Export wizard for MP4, ZIP, PDF. Mantine + Tailwind + Recharts.  

2. **Backend (FastAPI + Workers)**  
   - API Gateway (FastAPI REST /v1, Problem+JSON, RBAC).  
   - Workers: narrative, visual, audio, optimizer, consistency, safety, export.  
   - Event-driven via NATS + Redis Streams.  
   - Postgres + pgvector, S3/R2, Redis, OpenSearch.  

3. **Orchestration (CrewAI + LangGraph)**  
   - Agents as nodes: Narrative, Visual, Audio, Prompt Optimizer, Consistency Checker, Safety, Exporter.  
   - State tracked in Postgres, embeddings in pgvector.  
   - Guardrails: token/step budgets, brand kits, “facts first” stage before freeform generation.  

4. **DevOps & Security**  
   - Vercel for FE; Render/Fly/GKE for workers.  
   - CI/CD: GitHub Actions with linting, tests, Docker scan, blue/green deploy.  
   - IaC: Terraform for DB/Search/Redis/NATS/buckets.  
   - KMS-encrypted secrets, tenant-scoped brand kits, RLS everywhere.

### Success Criteria
**Product KPIs**
- ≥85% of packs pass alignment & safety on first try.  
- Time-to-publish < 10 min from prompt to MP4.  
- CSAT ≥ 4.5/5; repeat use ≥ 60% weekly.  
- Cost per pack reduced ≥25% after optimizer warm-up.  

**Engineering SLOs**
- Story Pack p95 (script+2 imgs+60s audio) < 45s.  
- Export MP4 < 12s p95.  
- Pipeline success ≥ 99% monthly.  
- Data freshness for brand kits immediate; asset CDN hit-rate ≥ 80%.  

### Rollout Plan
- **MVP**: Story Packs + Brand Kits + MP4 export.  
- **Phase 2**: Campaign Kits + Lesson Modules + ZIP/PDF bundles.  
- **Phase 3**: Presets, schedules, A/B automation.  
- **Phase 4**: Governance (review states, audit logs).  
