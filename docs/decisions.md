# Architecture Decision Records (ADR)

## ADR-001: Single-Machine Multi-Process Simulation

**Status**: Accepted
**Date**: 2026-05-04

**Context**: Need to simulate federated learning with multiple clients without deploying real distributed infrastructure.

**Decision**: Use Python `multiprocessing` to simulate clients on a single machine.

**Consequences**:
- (+) Focus on algorithm research, not distributed systems engineering
- (+) Easy debugging and reproducibility
- (-) Cannot test real network effects or heterogeneous hardware

---

## ADR-002: Frozen Backbone + Trainable PRM Head

**Status**: Accepted
**Date**: 2026-05-04

**Context**: Limited GPU resources (Pythia 1.4B for M2-M4).

**Decision**: Freeze the LLM backbone and only train a lightweight MLP head for step reward prediction.

**Consequences**:
- (+) Dramatically reduces training cost
- (+) Faster experimentation cycles
- (-) May limit PRM expressiveness compared to full fine-tuning
- (-) May need to revisit for M5 (LLaMA-3.1 8B)

---

## ADR-003: Client-Side DP-SGD

**Status**: Accepted
**Date**: 2026-05-04

**Context**: Step-level labels are the primary privacy concern in federated PRM training.

**Decision**: Apply DP-SGD during client local training, not at server aggregation.

**Consequences**:
- (+) Privacy budget is naturally per-client
- (+) Server never sees raw gradients from sensitive data
- (-) Cannot leverage secure aggregation for additional protection

---

## ADR-004: Step as Text Segment (Not Token Boundary)

**Status**: Accepted
**Date**: 2026-05-04

**Context**: Need a consistent definition of "step" across datasets.

**Decision**: A step is a text segment separated by `\n\n` or `####`. PRM processes the full step text including question context.

**Consequences**:
- (+) Compatible with PRM800K and ProcessBench
- (+) Human-interpretable step boundaries
- (-) Some datasets may require custom delimiters
