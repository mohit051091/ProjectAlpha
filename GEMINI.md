GLOBAL RULE FOR ALL PROJECTS:
1. At the start of any task, check if `graphify` is installed globally. If not, install it.
2. Before reading or editing any source code files, always read/check the `graphify-out/GRAPH_REPORT.md` first to locate files, map communities, and identify node dependencies, thereby optimizing token usage.
3. After every conversation, code change, or major task completion, you MUST run `graphify update .` in the root of the project to ensure documentation is always up to date.

# UNIVERSAL PROJECT GOVERNANCE RULES (PROJECT OS)

Every project must maintain a living project memory under a `/docs/` directory:

1. **Project Memory (`/docs/PROJECT_PLAYBOOK.md`):** Explains what the project is, why it exists, current architecture, and status.
2. **Decision Log (`/docs/DECISION_LOG.md`):** Records every significant technical decision with Date, Decision, Reason, Alternatives, Tradeoffs, and Outcome.
3. **Project Timeline (`/docs/PROJECT_TIMELINE.md`):** Chronological history of events (what changed, when, why, who/what triggered it, and the result).
4. **Incident & RCA Registry (`/docs/INCIDENTS.md`):** Logs every bug, outage, and production incident with Symptoms, Root Cause, Fix, and Preventive Actions.
5. **Task Registry (`/docs/TASKS.md`):** Tracks Backlog, Planned, In Progress, Testing, Ready, Deployed, and Cancelled tasks.
6. **Deployment Registry (`/docs/DEPLOYMENTS.md`):** Logs deployments with Date, Version, Changes, Risk, Rollback Plan, and Result.
7. **Architecture Blueprint (`/docs/ARCHITECTURE.md`):** Describes subsystems, data flows, dependencies, external services, and constraints.
8. **Current State Snapshot (`/docs/CURRENT_STATE.md`):** Instant overview of bottlenecks, priorities, open risks, pending deployments, known issues, and next steps.
9. **Optimization Registry (`/docs/PERFORMANCE_LOG.md`):** Tracks optimization attempts (Problem, Baseline, Hypothesis, Change, Result, and Acceptance).
10. **Rejected Ideas Registry (`/docs/REJECTED_APPROACHES.md`):** Logs rejected approaches and the reasons why they failed to prevent future repeated proposals.

## Mandatory Coding Agent Workflow Checklist:
- **Rule A:** Before making code changes, always review:
  * `docs/PROJECT_PLAYBOOK.md`
  * `docs/CURRENT_STATE.md`
  * `docs/TASKS.md`
  * `docs/DECISION_LOG.md`
  * `docs/ARCHITECTURE.md`
- **Rule B:** Every significant change must update:
  * Decision Log
  * Task Registry
  * Timeline
  * Current State
- **Rule C:** Every production bug must create or update:
  * Incident Record / Root Cause Analysis (RCA) / Fix Record
- **Rule D:** Every deployment must be recorded with Date, Version, Changes, and Outcome.
- **Rule E:** Every optimization must record Baseline, Hypothesis, Result, and Acceptance Decision.
- **Rule F:** Rejected approaches must be documented to prevent repeated proposals.
- **Rule G:** Documentation is part of the codebase and must evolve with the project. After every change, synchronize all project documentation and knowledge artifacts.
