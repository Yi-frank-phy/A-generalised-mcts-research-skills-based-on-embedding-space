# Hooks

Hooks are guardrails, not the main architecture.

Recommended hook points:

1. after executor episode: validate returned SearchNode / evidence / counterexample;
2. before final answer: verify DTE synthesis was produced;
3. before budget escalation: require explicit user approval;
4. after run: archive run spec and synthesis report.

Do not put the full DTE search engine inside a hook. Use hooks to enforce boundaries.
