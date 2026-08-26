"""One-shot fail-closed codemod for removing generic Judge production surface.

Phase 1 already stopped App-native runs from scheduling Judge.  This phase removes
Judge from the production protocol and all production entrypoints.  Persisted
artifacts that still contain role=judge or judge_payload intentionally fail the
current strict schema instead of keeping a hidden compatibility path alive.

The script is public and intentionally conservative: named top-level objects are
removed through Python AST source ranges, all structural text edits are asserted,
and every edited Python file is parsed again before anything is considered done.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "dte_backend"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def write(relative: str, text: str) -> None:
    path = ROOT / relative
    if path.suffix == ".py":
        ast.parse(text, filename=str(path))
    path.write_text(text, encoding="utf-8")


def require_count(text: str, needle: str, count: int, label: str) -> None:
    actual = text.count(needle)
    if actual != count:
        raise RuntimeError(f"{label}: expected {count} matches, found {actual}")


def replace_exact(text: str, old: str, new: str, label: str, *, count: int = 1) -> str:
    require_count(text, old, count, label)
    return text.replace(old, new, count)


def delete_between(text: str, start: str, end: str, label: str) -> str:
    require_count(text, start, 1, f"{label} start")
    if end not in text:
        raise RuntimeError(f"{label} end: marker not found")
    i = text.index(start)
    j = text.index(end, i)
    if j <= i:
        raise RuntimeError(f"{label}: invalid marker order")
    return text[:i] + text[j:]


def remove_top_levels(text: str, names: set[str], label: str) -> str:
    tree = ast.parse(text)
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            if node.name in found:
                raise RuntimeError(f"{label}: duplicate top-level {node.name}")
            found[node.name] = node
    missing = names - set(found)
    if missing:
        raise RuntimeError(f"{label}: missing top-level objects {sorted(missing)}")
    lines = text.splitlines(keepends=True)
    ranges = sorted(
        ((min([node.lineno, *[item.lineno for item in getattr(node, "decorator_list", [])]]) - 1, node.end_lineno or node.lineno) for node in found.values()),
        reverse=True,
    )
    for start, end in ranges:
        del lines[start:end]
    result = "".join(lines)
    ast.parse(result)
    return result


def remove_imported_names(text: str, module: str, names: set[str], label: str) -> str:
    tree = ast.parse(text)
    candidates = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and ("." * node.level + (node.module or "")) == module
        and names.intersection(alias.name for alias in node.names)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{label}: expected one import from {module}, found {len(candidates)}")
    node = candidates[0]
    remaining = [alias for alias in node.names if alias.name not in names]
    if any(alias.asname for alias in remaining):
        rendered = ", ".join(
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in remaining
        )
    else:
        rendered = ", ".join(alias.name for alias in remaining)
    replacement = "" if not remaining else f"from {module} import {rendered}\n"
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    lines[start:end] = [replacement] if replacement else []
    result = "".join(lines)
    ast.parse(result)
    return result


def remove_standalone_if_by_test(text: str, test_fragment: str, label: str) -> str:
    tree = ast.parse(text)
    matches: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.orelse:
            continue
        segment = ast.get_source_segment(text, node.test) or ""
        if test_fragment in segment:
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected one standalone if, found {len(matches)}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    del lines[node.lineno - 1 : node.end_lineno or node.lineno]
    result = "".join(lines)
    ast.parse(result)
    return result


def migrate_episode_models() -> None:
    rel = "src/dte_backend/episode_models.py"
    text = read(rel)
    if "JudgeEpisodeOutput" not in text and '"judge"' not in text:
        return
    text = replace_exact(
        text,
        'EpisodeRole = Literal["executor", "seed", "judge", "relation", "synthesis"]',
        'EpisodeRole = Literal["executor", "seed", "relation", "synthesis"]',
        "EpisodeRole vocabulary",
    )
    text = remove_top_levels(
        text,
        {"JudgeNodeInput", "JudgeEpisodePayload", "JudgeObservation", "JudgeEpisodeOutput"},
        "Judge episode models",
    )
    text = replace_exact(
        text,
        "    judge_payload: JudgeEpisodePayload | None = None\n",
        "",
        "EpisodeRequest Judge payload",
    )
    text = delete_between(
        text,
        '        elif self.role == "judge":\n',
        '        elif self.role == "relation":\n',
        "EpisodeRequest Judge validator branch",
    )
    snippet = (
        '            if self.judge_payload is not None:\n'
        '                raise ValueError("judge_payload requires role=\'judge\'")\n'
    )
    text = replace_exact(text, snippet, "", "role-specific Judge payload guards", count=2)
    text = replace_exact(
        text,
        '        elif self.judge_payload is not None:\n            raise ValueError("judge_payload requires role=\'judge\'")\n',
        "",
        "generic Judge payload guard",
    )
    text = replace_exact(
        text,
        "    structured_output: ExecutorEpisodeOutput | JudgeEpisodeOutput | RelationEpisodeOutput | None\n",
        "    structured_output: ExecutorEpisodeOutput | RelationEpisodeOutput | None\n",
        "EpisodeResult output union",
    )
    text = replace_exact(
        text,
        "    output: ExecutorEpisodeOutput | JudgeEpisodeOutput | RelationEpisodeOutput | None,\n",
        "    output: ExecutorEpisodeOutput | RelationEpisodeOutput | None,\n",
        "compute_output_hash union",
    )
    write(rel, text)


def migrate_models() -> None:
    rel = "src/dte_backend/models.py"
    text = read(rel)
    for line in (
        "    judge_reasoning: str | None = None\n",
        "    judge_risks: list[str] = Field(default_factory=list)\n",
        "    judge_uncertainty_evidence: list[str] = Field(default_factory=list)\n",
        "    judge_result_provenance: dict[str, str] | None = None\n",
    ):
        if line in text:
            text = text.replace(line, "", 1)
    write(rel, text)


def migrate_relation_models() -> None:
    rel = "src/dte_backend/relation_models.py"
    text = read(rel)
    for line in (
        "    judge_reasoning: str | None = None\n",
        "    judge_risks: list[str] = Field(default_factory=list)\n",
        "    judge_uncertainty_evidence: list[str] = Field(default_factory=list)\n",
        "    judge_result_provenance: dict[str, str] | None = None\n",
    ):
        if line in text:
            text = text.replace(line, "", 1)
    text = text.replace(
        '    """Oracle-visible node material with no Judge or selection metadata."""',
        '    """Oracle-visible node material with no controller selection metadata."""',
    )
    write(rel, text)


def migrate_episode_adapter() -> None:
    rel = "src/dte_backend/episode_adapter.py"
    text = read(rel)
    if "build_judge_episode_request" not in text:
        return
    text = remove_imported_names(
        text,
        ".episode_models",
        {"JudgeEpisodePayload", "JudgeNodeInput"},
        "episode_adapter Judge imports",
    )
    text = remove_top_levels(text, {"build_judge_episode_request"}, "Judge request builder")
    for field in (
        '            "judge_reasoning",\n',
        '            "judge_risks",\n',
        '            "judge_uncertainty_evidence",\n',
        '            "judge_result_provenance",\n',
    ):
        text = text.replace(field, "")
    legacy = (
        "        judge_reasoning=node.judge_reasoning,\n"
        "        judge_risks=list(node.judge_risks),\n"
        "        judge_uncertainty_evidence=list(node.judge_uncertainty_evidence),\n"
        "        judge_result_provenance=(\n"
        "            None\n"
        "            if node.judge_result_provenance is None\n"
        "            else dict(node.judge_result_provenance)\n"
        "        ),\n"
    )
    text = replace_exact(text, legacy, "", "legacy Relation Judge metadata")
    write(rel, text)


def migrate_runner() -> None:
    rel = "src/dte_backend/runner.py"
    text = read(rel)
    if "_validated_judge_results" not in text:
        return
    text = remove_imported_names(text, ".judge", {"batch_judge"}, "runner Judge implementation import")
    text = remove_imported_names(text, ".oracle_validation", {"validate_judge_output"}, "runner Judge validation import")
    text = remove_imported_names(text, ".subprocess_oracles", {"JudgeAdapter"}, "runner Judge adapter import")
    text = remove_top_levels(text, {"_validated_judge_results"}, "runner Judge helper")
    text = replace_exact(text, "    judge_adapter: JudgeAdapter | None = None,\n", "", "runner Judge parameter")
    text = text.replace(
        "role-isolated seeding -> Judge -> embedding/KDE/entropy -> UCB/Boltzmann ->\n",
        "role-isolated seeding -> embedding/KDE/entropy -> UCB/Boltzmann ->\n",
    )
    text = delete_between(
        text,
        "        judge_results = _validated_judge_results(frontier, judge_adapter, cache)\n",
        "        frontier, kde_state = estimate_frontier_kde_state(\n",
        "runner per-iteration Judge stage",
    )
    text = text.replace('                    f"judge_cache_hits={cache.stats.judge_hits}",\n', "")
    write(rel, text)


def migrate_strict_runner() -> None:
    rel = "src/dte_backend/strict_runner.py"
    text = read(rel)
    if "judge_command" not in text and "JudgeAdapter" not in text:
        return
    text = remove_imported_names(text, ".subprocess_oracles", {"JudgeAdapter"}, "strict runner Judge import")
    text = text.replace("    allow_heuristic_judge: bool\n", "")
    text = text.replace("            allow_heuristic_judge=True,\n", "")
    text = text.replace("            allow_heuristic_judge=False,\n", "")
    text = text.replace('        "examples/mock_judge_adapter.py",\n', "")
    text = text.replace("    judge_command: str | None,\n", "")
    text = delete_between(
        text,
        '    if not policy.allow_heuristic_judge and not judge_command:\n',
        '    if _is_mock_command(executor_command) and not policy.allow_mock:\n',
        "strict Judge policy checks",
    )
    text = text.replace("    judge_adapter: JudgeAdapter | None = None,\n", "")
    text = text.replace("    judge_command: str | None = None,\n", "")
    text = text.replace("        judge_command=judge_command,\n", "")
    text = text.replace("        judge_adapter=judge_adapter,\n", "")
    write(rel, text)


def migrate_cli() -> None:
    rel = "src/dte_backend/__main__.py"
    text = read(rel)
    if "judge-oracle" not in text and "judge_command" not in text:
        return
    text = remove_imported_names(
        text,
        ".subprocess_oracles",
        {"build_subprocess_judge_adapter", "run_subprocess_judge"},
        "CLI Judge imports",
    )
    text = remove_top_levels(text, {"cmd_judge_oracle"}, "CLI Judge oracle command")
    run_block = (
        "    judge_adapter = None\n"
        "    if args.judge_command:\n"
        "        judge_adapter = build_subprocess_judge_adapter(split_command(args.judge_command), timeout=args.judge_timeout)\n"
    )
    text = replace_exact(text, run_block, "", "CLI Judge adapter blocks", count=2)
    text = text.replace("        judge_adapter=judge_adapter,\n", "")
    text = text.replace("            judge_adapter=judge_adapter,\n", "")
    text = text.replace("            judge_command=args.judge_command,\n", "")
    text = delete_between(
        text,
        '    judge = sub.add_parser("judge-oracle", help="run and validate a Judge oracle command")\n',
        '    relation = sub.add_parser("relation-oracle", help="run and validate a relation oracle command")\n',
        "CLI Judge subcommand",
    )
    for line in (
        '    run.add_argument("--judge-command", help="optional subprocess Judge oracle command")\n',
        '    run.add_argument("--judge-timeout", type=float, default=360.0)\n',
        '    strict.add_argument("--judge-command", help="required in real mode")\n',
        '    strict.add_argument("--judge-timeout", type=float, default=360.0)\n',
    ):
        text = replace_exact(text, line, "", f"CLI option {line.strip()}")
    write(rel, text)


def migrate_subprocess_oracles() -> None:
    rel = "src/dte_backend/subprocess_oracles.py"
    text = read(rel)
    if "JudgeAdapter" not in text:
        return
    text = remove_imported_names(text, ".oracle_validation", {"validate_judge_output"}, "subprocess Judge validator import")
    text = remove_imported_names(text, ".oracles", {"JudgeOracleResult", "make_judge_task"}, "subprocess Judge task imports")
    text = text.replace("JudgeAdapter = Callable[[list[SearchNode]], list[JudgeOracleResult]]\n", "")
    text = remove_top_levels(text, {"run_subprocess_judge", "build_subprocess_judge_adapter"}, "subprocess Judge functions")
    write(rel, text)


def migrate_oracles() -> None:
    rel = "src/dte_backend/oracles.py"
    text = read(rel)
    if "JudgeOracleResult" not in text:
        return
    text = remove_top_levels(text, {"JudgeOracleResult", "make_judge_task"}, "legacy Judge oracle objects")
    text = replace_exact(
        text,
        'OracleKind = Literal["judge", "equivalent_merge", "complementary_merge", "conflict_merge", "discriminator"]',
        'OracleKind = Literal["equivalent_merge", "complementary_merge", "conflict_merge", "discriminator"]',
        "OracleKind Judge member",
    )
    text = text.replace(
        "Judge, complementary merge, conflict merge, and discriminator generation are not\n",
        "Complementary merge, conflict merge, and discriminator generation are not\n",
    )
    write(rel, text)


def migrate_oracle_validation() -> None:
    rel = "src/dte_backend/oracle_validation.py"
    text = read(rel)
    if "validate_judge_output" not in text:
        return
    text = remove_imported_names(text, ".oracles", {"JudgeOracleResult"}, "Judge validation result import")
    text = remove_top_levels(text, {"validate_judge_output"}, "Judge validator")
    text = text.replace(
        "Judge and relation oracles may be performed by subagents. The backend validates\n",
        "Relation oracles may be performed by subagents. The backend validates\n",
    )
    write(rel, text)


def migrate_episode_commit() -> None:
    rel = "src/dte_backend/episode_commit.py"
    text = read(rel)
    if 'request.role == "judge"' not in text:
        return
    text = remove_imported_names(text, ".episode_models", {"JudgeEpisodeOutput"}, "commit Judge output import")
    text = remove_top_levels(text, {"_raw_judge_observations"}, "raw Judge observation helper")
    text = replace_exact(
        text,
        "    raw_nodes = _raw_nodes(raw_result) or _raw_judge_observations(raw_result) or _raw_relation_observations(raw_result)\n",
        "    raw_nodes = _raw_nodes(raw_result) or _raw_relation_observations(raw_result)\n",
        "quality-count Judge observations",
    )
    text = text.replace('0 if request.role in {"judge", "relation"} else None', '0 if request.role == "relation" else None')
    text = text.replace('0 if role in {"judge", "relation"} else None', '0 if role == "relation" else None')
    raw_block = (
        "        raw_observations = (\n"
        "            _raw_judge_observations(result_payload)\n"
        "            if request.role == \"judge\"\n"
        "            else _raw_relation_observations(result_payload)\n"
        "            if request.role == \"relation\"\n"
        "            else []\n"
        "        )\n"
        "        returned_observation_count = len(raw_observations) if request.role in {\"judge\", \"relation\"} else None\n"
    )
    text = replace_exact(
        text,
        raw_block,
        '        raw_observations = _raw_relation_observations(result_payload) if request.role == "relation" else []\n'
        '        returned_observation_count = len(raw_observations) if request.role == "relation" else None\n',
        "commit raw Judge observation dispatch",
    )
    text = delete_between(
        text,
        '    if request.role == "judge":\n',
        '    if request.role == "relation":\n',
        "Judge commit transaction",
    )
    write(rel, text)


def migrate_app_driver() -> None:
    rel = "src/dte_backend/app_driver.py"
    text = read(rel)
    if '"judge"' not in text and "JudgeEpisodeOutput" not in text:
        return
    text = remove_imported_names(text, ".episode_adapter", {"build_judge_episode_request"}, "App Judge builder import")
    text = remove_imported_names(text, ".episode_models", {"JudgeEpisodeOutput"}, "App Judge output import")
    text = text.replace('Literal["executor", "seed", "judge", "relation", "synthesis"]', 'Literal["executor", "seed", "relation", "synthesis"]')
    text = text.replace("isinstance(output, (ExecutorEpisodeOutput, JudgeEpisodeOutput))", "isinstance(output, ExecutorEpisodeOutput)")
    text = remove_standalone_if_by_test(
        text,
        'episode.role == "judge"',
        "App committed Judge validation branch",
    )
    text = delete_between(
        text,
        "    judge_observations: dict[\n",
        "    allocation_record_by_node = {\n",
        "App persisted Judge observation index",
    )
    text = delete_between(
        text,
        '    elif previous.request.role == "judge":\n',
        '    elif previous.request.role == "relation":\n',
        "App legacy Judge retry",
    )
    # Once Judge persistence is gone, node-state validation is based only on
    # geometry/controller facts; any old artifact carrying removed fields fails
    # SearchNode/EpisodeRequest schema validation before reaching this point.
    text = text.replace("judge_observations.get(node.node_id)", "None")
    write(rel, text)


def migrate_cache() -> None:
    rel = "src/dte_backend/cache.py"
    text = read(rel)
    if "JudgeCacheNamespace" not in text:
        return
    text = text.replace("from .context_envelope import evaluation_payload, semantic_embedding_payload\n", "from .context_envelope import semantic_embedding_payload\n")
    text = remove_top_levels(
        text,
        {"JudgeCacheNamespace", "JudgeCacheEntry", "judge_cache_key", "stable_node_payload", "stable_node_hash"},
        "Judge cache objects",
    )
    text = delete_between(
        text,
        "DEFAULT_JUDGE_NAMESPACE = JudgeCacheNamespace(\n",
        "def embedding_cache_key(\n",
        "default Judge cache namespace",
    )
    text = text.replace("    judge_hits: int = 0\n", "")
    text = text.replace("    judge_misses: int = 0\n", "")
    text = text.replace("    judge_scores: dict[str, JudgeCacheEntry] = field(default_factory=dict)\n", "")
    text = remove_top_levels(text, set(), "noop") if False else text
    # Methods are class members, so remove their source ranges by method name.
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DTECache")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"get_judge", "set_judge"}]
    if len(methods) != 2:
        raise RuntimeError(f"Judge cache methods: expected 2, found {len(methods)}")
    lines = text.splitlines(keepends=True)
    for node in sorted(methods, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1 : node.end_lineno or node.lineno]
    text = "".join(lines)
    write(rel, text)


def migrate_file_cache() -> None:
    rel = "src/dte_backend/file_cache.py"
    text = read(rel)
    if "JudgeCacheNamespace" not in text:
        return
    text = remove_imported_names(
        text,
        ".cache",
        {"DEFAULT_JUDGE_NAMESPACE", "JudgeCacheEntry", "JudgeCacheNamespace", "judge_cache_key"},
        "file cache Judge imports",
    )
    text = text.replace('    """Simple JSON-backed cache with split embedding/Judge identities."""', '    """Simple JSON-backed embedding cache."""')
    text = text.replace('        self.data = {"vectors": {}, "scores": {}}\n', '        self.data = {"vectors": {}}\n')
    text = text.replace('            self.data.setdefault("scores", {})\n', "")
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FileDTECache")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"get_judge", "set_judge"}]
    if len(methods) != 2:
        raise RuntimeError(f"file Judge cache methods: expected 2, found {len(methods)}")
    lines = text.splitlines(keepends=True)
    for node in sorted(methods, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1 : node.end_lineno or node.lineno]
    text = "".join(lines)
    write(rel, text)


def delete_judge_files() -> None:
    for relative in (
        "src/dte_backend/judge.py",
        "scripts/codex_judge_adapter.py",
        "prompts/judge.md",
        "prompts/judge_oracle.md",
        "examples/mock_judge_adapter.py",
        "examples/subagent_transcripts/judge_call.json",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def assert_phase2_invariants() -> None:
    checks = {
        "src/dte_backend/episode_models.py": ["JudgeEpisode", '"judge"'],
        "src/dte_backend/episode_adapter.py": ["build_judge_episode_request", "JudgeNodeInput"],
        "src/dte_backend/runner.py": ["judge_adapter", "_validated_judge_results", "batch_judge"],
        "src/dte_backend/strict_runner.py": ["judge_command", "JudgeAdapter", "allow_heuristic_judge"],
        "src/dte_backend/__main__.py": ["judge-oracle", "judge_command", "build_subprocess_judge_adapter"],
        "src/dte_backend/subprocess_oracles.py": ["JudgeAdapter", "run_subprocess_judge", "JudgeOracleResult"],
        "src/dte_backend/oracles.py": ["JudgeOracleResult", "make_judge_task", '"judge"'],
        "src/dte_backend/oracle_validation.py": ["validate_judge_output", "JudgeOracleResult"],
        "src/dte_backend/models.py": ["judge_reasoning", "judge_risks", "judge_uncertainty_evidence", "judge_result_provenance"],
        "src/dte_backend/relation_models.py": ["judge_reasoning", "judge_risks", "judge_uncertainty_evidence", "judge_result_provenance"],
        "src/dte_backend/cache.py": ["JudgeCache", "judge_cache", "judge_scores", "judge_hits", "judge_misses"],
        "src/dte_backend/file_cache.py": ["JudgeCache", "judge_cache", "get_judge", "set_judge"],
    }
    failures: list[str] = []
    for relative, needles in checks.items():
        text = read(relative)
        for needle in needles:
            if needle in text:
                failures.append(f"{relative}: {needle}")
        ast.parse(text, filename=relative)
    for relative in (
        "src/dte_backend/judge.py",
        "scripts/codex_judge_adapter.py",
        "prompts/judge.md",
        "prompts/judge_oracle.md",
        "examples/mock_judge_adapter.py",
    ):
        if (ROOT / relative).exists():
            failures.append(f"file remains: {relative}")
    if failures:
        raise RuntimeError("phase-2 Judge invariants failed:\n" + "\n".join(failures))


def main() -> None:
    migrate_episode_models()
    migrate_models()
    migrate_relation_models()
    migrate_episode_adapter()
    migrate_runner()
    migrate_strict_runner()
    migrate_cli()
    migrate_subprocess_oracles()
    migrate_oracles()
    migrate_oracle_validation()
    migrate_episode_commit()
    migrate_app_driver()
    migrate_cache()
    migrate_file_cache()
    delete_judge_files()
    assert_phase2_invariants()
    print("generic Judge phase-2 production removal applied")


if __name__ == "__main__":
    main()
