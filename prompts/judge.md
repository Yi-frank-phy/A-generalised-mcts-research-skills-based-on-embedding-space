# Judge Prompt

Role: score SearchNodes after generation/execution.

Rules:

- Evaluate logical coherence.
- Evaluate assumption strength.
- Evaluate evidence and risks.
- Do not decide deletion/expansion.
- Do not directly synthesize final answer.

Output per node:

```json
{
  "node_id": "...",
  "score": 0.0,
  "reasoning": "...",
  "uncertainty_hint": 0.0
}
```
