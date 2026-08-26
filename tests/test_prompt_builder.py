from dte_backend.prompt_builder import build_cached_subagent_prompt, load_static_prefix, prompts_dir




def test_prompt_builder_places_dynamic_payload_last():
    prompt = build_cached_subagent_prompt("relation", {"z": 1, "a": 2})
    assert "# Dynamic task input" in prompt
    assert prompt.rstrip().endswith("```")
    assert prompt.index("# Dynamic task input") > prompt.index("# Relation Oracle Subagent Prompt")




