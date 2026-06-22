# Product Requirements Document: DTE Codex Skill Backend

## 1. Problem

The existing DTE system has a valuable mathematical search engine based on frontier search, role-isolated evaluation, entropy/novelty guidance, UCB, and Boltzmann expansion. However, running the full agent pipeline can be slow and expensive due to many LLM round-trips and local computation.

The user wants to preserve the mandatory DTE flow while using Codex-level high quota and strong reasoning/execution as a backend.

## 2. Goal

Build a minimal backend/skill package that lets Codex or similar agents run DTE as a fixed research protocol.

The architecture must not be redesigned per task. The user should only provide task parameters.

## 3. Non-goals

- Not a SaaS.
- Not a pure self-organized swarm.
- Not a LangChain-first rewrite.
- Not a Markdown-driven architecture generator.
- Not a quota resale or platform bypass tool.

## 4. Users

Primary user: one researcher/student using DTE for high-depth personal academic and conceptual research.

## 5. Core requirements

### R1: Mandatory DTE flow
All final research outputs must pass through DTE-controlled Judge/Evolution/Synthesis.

### R2: Role separation as anti-bias isolation
Generation, judgement, execution, and synthesis remain logically separate.

### R3: Structured interface
All machine-facing inputs and outputs use schemas. Free-form Markdown can only be a human-facing shell.

### R4: Codex/Kimi/OpenClaw executor compatibility
Executor episodes can use external coding/research agents, but must submit structured node outputs.

### R5: Low round-trip design
The system should batch and cache where possible while preserving logical roles.

### R6: Hard budget caps
Budget controls are mandatory. Exploration is bounded by hard limits rather than by changing UCB by default.

## 6. Success criteria

- A Codex agent can call the DTE skill with a run spec.
- The backend validates SearchNode outputs.
- The backend can allocate expansion budgets from node scores and embeddings/features.
- The system can produce a synthesis report with uncertainty and rejected alternatives.
- A run can complete with a small default budget: 2 iterations, 3 child budget.
