from dte_backend.prompt_builder import build_cached_subagent_prompt, load_static_prefix


def test_prompt_builder_places_static_prefix_first():
    prompt = build_cached_subagent_prompt("judge", {"task": "dynamic", "nodes": []})
    prefix = load_static_prefix()
    assert prompt.startswith(prefix)


def test_prompt_builder_places_dynamic_payload_last():
    prompt = build_cached_subagent_prompt("relation", {"z": 1, "a": 2})
    assert "# Dynamic task input" in prompt
    assert prompt.rstrip().endswith("```")
    assert prompt.index("# Dynamic task input") > prompt.index("# Relation Oracle Subagent Prompt")


def test_prompt_builder_shares_prefix_across_roles():
    judge = build_cached_subagent_prompt("judge", {"task": "j"})
    executor = build_cached_subagent_prompt("executor", {"task": "e"})
    prefix = load_static_prefix()
    assert judge[: len(prefix)] == executor[: len(prefix)]
