"""Thin command-line interface for every roadmap phase."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import NoReturn

from efficient_llm.core.config import ConfigError, config_as_yaml, load_config
from efficient_llm.core.hardware import collect_environment
from efficient_llm.core.paths import resolve_project_path
from efficient_llm.core.reproducibility import OutputExistsError, atomic_write_json

EXIT_CONFIG = 2
EXIT_ENVIRONMENT = 3
EXIT_BACKEND = 4


def _add_execution_options(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    if config:
        parser.add_argument("--config", required=True, help="project-relative YAML configuration")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print without computing"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)


def _nested(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], name: str
) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    parent = subparsers.add_parser(name)
    return parent.add_subparsers(dest=f"{name}_command", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efficient-llm", description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="capture local hardware and environment metadata")
    doctor.add_argument("--output", required=True)
    doctor.add_argument("--dry-run", action="store_true")

    data = _nested(commands, "data")
    prepare = data.add_parser("prepare")
    _add_execution_options(prepare)
    inspect = data.add_parser("inspect")
    inspect.add_argument("--manifest", required=True)
    _add_execution_options(inspect, config=False)

    mini = _nested(commands, "mini")
    _add_execution_options(mini.add_parser("train"))

    evaluate = _nested(commands, "evaluate")
    for name in ("math", "regression"):
        command = evaluate.add_parser(name)
        _add_execution_options(command)
        command.add_argument("--model")

    _add_execution_options(commands.add_parser("train"))

    select = commands.add_parser("select")
    select.add_argument("--runs", nargs="+", required=True)
    select.add_argument("--output", required=True)
    _add_execution_options(select, config=False)

    merge = commands.add_parser("merge")
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--output", required=True)
    _add_execution_options(merge, config=False)

    quantize = _nested(commands, "quantize")
    for name in ("probe", "run"):
        _add_execution_options(quantize.add_parser(name))

    serve = _nested(commands, "serve")
    for name in ("hf", "vllm"):
        _add_execution_options(serve.add_parser(name))

    kv_cache = commands.add_parser("kv-cache")
    kv_cache.add_argument("--model-config", required=True)
    kv_cache.add_argument("--seq-len", required=True, type=int)
    kv_cache.add_argument("--batch-size", required=True, type=int)
    kv_cache.add_argument("--dtype", required=True, choices=("bf16", "fp16", "fp32"))
    _add_execution_options(kv_cache, config=False)

    benchmark = _nested(commands, "benchmark")
    online = benchmark.add_parser("online")
    _add_execution_options(online)
    online.add_argument("--endpoint", required=True)

    results = _nested(commands, "results")
    aggregate = results.add_parser("aggregate")
    aggregate.add_argument("--results", required=True)
    _add_execution_options(aggregate, config=False)
    plot = results.add_parser("plot")
    plot.add_argument("--summaries", required=True)
    _add_execution_options(plot, config=False)
    return parser


def _fail(message: str, code: int) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _dry_run(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", None)
    payload: dict[str, object] = {
        "status": "dry-run",
        "command": args.command,
        "arguments": {key: value for key, value in vars(args).items() if key != "dry_run"},
    }
    if config_path:
        loaded = load_config(config_path)
        payload["config_source"] = str(loaded.source)
        payload["config_hash"] = loaded.config_hash
        payload["resolved_config"] = loaded.resolved
        payload["resolved_config_yaml"] = config_as_yaml(loaded)
        payload["resolved_output_dir"] = str(
            resolve_project_path(loaded.model.run.output_dir)
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    output = resolve_project_path(args.output)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "output": str(output)}, indent=2))
        return 0
    if output.exists():
        raise OutputExistsError(f"refusing to overwrite existing environment report: {output}")
    report = collect_environment()
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _run_doctor(args)
        if args.dry_run:
            return _dry_run(args)
        _fail(
            f"'{args.command}' is a roadmap placeholder in Phase 0; use --dry-run or "
            "implement its owning phase first",
            EXIT_BACKEND,
        )
    except (ConfigError, OutputExistsError, ValueError) as exc:
        _fail(str(exc), EXIT_CONFIG)


if __name__ == "__main__":
    main()
