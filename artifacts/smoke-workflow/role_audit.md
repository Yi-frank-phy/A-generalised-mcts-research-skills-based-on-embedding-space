# DTE Role Audit

This records the logical seed roles: decomposition, research, optional compile hint, and strategy generation.

```json
{
  "decomposition": {
    "core_question": "Explore whether a proposed high-dimensional quantum protocol can satisfy the required symmetry constraints",
    "subquestions": [
      "What are the minimal assumptions needed for: Explore whether a proposed high-dimensional quantum protocol can satisfy the required symmetry constraints",
      "Which constructive route can satisfy the goal: Produce a rigorous but concise research report with assumptions, derivation paths, rejected alternatives, and confidence levels",
      "Which counterexamples or boundary cases would refute the proposed route?",
      "Which branches are equivalent, complementary, or in conflict?"
    ],
    "constraints": [
      "Preserve mathematical derivations.",
      "Separate conjecture from proven claims.",
      "Do not bypass the DTE Judge/Evolution/Synthesis protocol."
    ]
  },
  "research_context": {
    "background": [
      "Core problem: Explore whether a proposed high-dimensional quantum protocol can satisfy the required symmetry constraints",
      "Goal: Produce a rigorous but concise research report with assumptions, derivation paths, rejected alternatives, and confidence levels."
    ],
    "unknowns": [
      "Whether the direct construction satisfies all constraints.",
      "Whether a counterexample exists in a boundary or low-dimensional case.",
      "Whether two promising branches are actually equivalent under a change of formalism."
    ],
    "failure_modes": [
      "Hidden assumption is stronger than stated constraints.",
      "Search branch is semantically redundant with an existing node.",
      "Executor produces an answer-like synthesis instead of structured evidence."
    ]
  },
  "compile_hint": {
    "summary_focus": "Compile only the information needed to create or evaluate SearchNodes.",
    "preserve": [
      "explicit assumptions",
      "evidence and counterexamples",
      "branch conflicts and merge opportunities",
      "uncertainty / failure modes"
    ],
    "drop": [
      "long transcripts",
      "irrelevant tool logs",
      "style-only rewrites",
      "self-justifying explanations without new evidence"
    ]
  },
  "distiller_role": "removed: compile is an optional agent-local instruction, not a mandatory backend step"
}
```
