#!/usr/bin/env python3
"""
Compatibility matrix runner for MRSegmentator.

Requires: uv  (pip install uv)

Usage:
    python tests/compatibility/run_matrix.py [--keep-envs]

Reads:   tests/compatibility/matrix.yaml
Writes:  tests/compatibility/results/<id>/run.log
         tests/compatibility/results/<id>/versions.json
         tests/compatibility/results/<id>/freeze.txt
         tests/compatibility/results/<id>/reports/
         tests/compatibility/summary.txt
         tests/compatibility/summary.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # .../mrsegmentator
COMPAT = ROOT / "tests" / "compatibility"
# ENV_DIR    = COMPAT / "envs"
ENV_DIR = Path("/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/kidney/temp_envs")
RESULT_DIR = COMPAT / "results"
MATRIX = COMPAT / "matrix.yaml"


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], logfile: Path, cwd: Path | None = None) -> int:
    """Run a command, appending stdout+stderr to logfile. Returns exit code."""
    with logfile.open("a") as f:
        f.write(f"\n{'=' * 72}\n{' '.join(str(c) for c in cmd)}\n{'=' * 72}\n")
        return subprocess.run(
            cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True
        ).returncode


def _check_output(cmd: list[str]) -> str:
    return subprocess.check_output([str(c) for c in cmd], text=True, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# uv-based environment lifecycle
# ---------------------------------------------------------------------------


def create_env(env: Path, py_version: str, logfile: Path) -> bool:
    """
    Create a venv via uv. Returns True on success.
    uv will download the requested Python if not present on the system.
    uv will download missing versions automatically; use --no-python-downloads to disable.
    """
    rc = _run(
        ["uv", "venv", "--python", py_version, str(env)],
        logfile,
    )
    return rc == 0


def install(cfg: dict, env: Path, logfile: Path) -> bool:
    """
    Pin torch / numpy / nnunetv2 explicitly, then install the package
    with --no-deps so pip cannot override our pins to satisfy setup.cfg.
    """
    py = env / "bin" / "python"
    steps = [
        ["uv", "pip", "install", "--python", str(py), "--upgrade", "pip"],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(py),
            f"torch=={cfg['torch']}",
            f"numpy=={cfg['numpy']}",
            f"nnunetv2=={cfg['nnunetv2']}",
        ],
        ["uv", "pip", "install", "--python", str(py), "--no-deps", "-e", str(ROOT)],
    ]
    return all(_run(cmd, logfile) == 0 for cmd in steps)


def collect_versions(env: Path, result: Path) -> dict:
    """
    Use importlib.metadata (works for all packages regardless of __version__).
    Also writes a full pip freeze for traceability.
    """
    py = env / "bin" / "python"
    code = (
        "import json, importlib.metadata as m, torch;"
        "print(json.dumps({"
        "'python': __import__('platform').python_version(),"
        "'torch':    m.version('torch'),"
        "'numpy':    m.version('numpy'),"
        "'nnunetv2': m.version('nnunetv2'),"
        "'cuda':     torch.version.cuda"
        "}))"
    )
    out = _check_output([py, "-c", code])
    versions = json.loads(out)
    (result / "versions.json").write_text(json.dumps(versions, indent=2))

    freeze = _check_output(["uv", "pip", "freeze", "--python", str(py)])
    (result / "freeze.txt").write_text(freeze)

    return versions


def run_tests(env: Path, logfile: Path) -> tuple[int, float]:
    env_vars = {**os.environ, "PATH": f"{env / 'bin'}:{os.environ['PATH']}"}
    t0 = time.perf_counter()
    with logfile.open("a") as f:
        rc = subprocess.run(
            ["make", "full"],
            cwd=ROOT,
            env=env_vars,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode
    return rc, time.perf_counter() - t0


def copy_reports(result: Path) -> None:
    src = ROOT / "reports"
    if src.exists():
        shutil.copytree(src, result / "reports", dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Version pin validation
# ---------------------------------------------------------------------------


def check_versions(cfg: dict, actual: dict) -> list[str]:
    """Return warnings where installed version doesn't start with requested pin."""
    warnings = []
    for key in ("torch", "numpy", "nnunetv2"):
        req = cfg[key]
        got = actual.get(key, "?")
        if not got.startswith(req):
            warnings.append(f"{key}: requested {req}, got {got}")
    return warnings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLS = (18, 8, 8, 7, 6, 7, 8)  # id  status  time  python  torch  numpy  nnunet


def _row(*cells) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, _COLS))


def _format_row(r: dict) -> str:
    return _row(
        r["id"],
        r["status"],
        f"{r['runtime']:.1f}s" if r["runtime"] is not None else "—",
        r["cfg"].get("python", "—"),
        r["cfg"].get("torch", "—"),
        r["cfg"].get("numpy", "—"),
        r["cfg"].get("nnunetv2", "—"),
    )


def print_summary(summary: list[dict]) -> None:
    header = _row("id", "status", "time", "python", "torch", "numpy", "nnunetv2")
    sep = "─" * len(header)
    lines = [
        "",
        sep,
        "Compatibility Summary",
        sep,
        header,
        sep,
        *[_format_row(r) for r in summary],
        sep,
    ]
    passed = sum(1 for r in summary if r["status"] == "PASS")
    lines.append(f"{passed}/{len(summary)} passed")
    lines.append(sep)
    print("\n".join(lines))


def write_summary(summary: list[dict]) -> None:
    header = _row("id", "status", "time", "python", "torch", "numpy", "nnunetv2")
    sep = "-" * len(header)
    rows = "\n".join(_format_row(r) for r in summary)
    passed = sum(1 for r in summary if r["status"] == "PASS")
    txt = f"{header}\n{sep}\n{rows}\n{sep}\n{passed}/{len(summary)} passed\n"
    (COMPAT / "summary.txt").write_text(txt)
    (COMPAT / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--keep-envs",
        action="store_true",
        help="Preserve venvs after each run (useful for debugging)",
    )
    args = parser.parse_args()

    ENV_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    matrix = yaml.safe_load(MATRIX.read_text())
    baseline = matrix.get("baseline", {})
    summary: list[dict] = []

    for test in matrix["tests"]:
        cfg = {**baseline, **test}
        label = test.get("id", str(cfg))
        print(f"\n[{label}]")

        env = ENV_DIR / label
        result = RESULT_DIR / label
        shutil.rmtree(result, ignore_errors=True)
        result.mkdir(parents=True)
        logfile = result / "run.log"

        if env.exists():
            shutil.rmtree(env)

        record: dict = {"id": label, "cfg": cfg, "status": "—", "runtime": None}

        # --- venv ---
        print("  creating venv ...", end=" ", flush=True)
        if not create_env(env, cfg["python"], logfile):
            print("SKIP  (Python not available)")
            record["status"] = "SKIP"
            summary.append(record)
            continue
        print("ok")

        # --- install ---
        print("  installing     ...", end=" ", flush=True)
        if not install(cfg, env, logfile):
            print("FAILED  (see run.log)")
            record["status"] = "FAILED(install)"
            summary.append(record)
            if not args.keep_envs:
                shutil.rmtree(env)
            continue
        print("ok")

        # --- version check ---
        try:
            versions = collect_versions(env, result)
            for w in check_versions(cfg, versions):
                print(f"  WARNING: {w}")
        except Exception as exc:
            print(f"  WARNING: could not collect versions: {exc}")
            versions = {}

        # --- tests ---
        print("  running tests  ...", end=" ", flush=True)
        rc, runtime = run_tests(env, logfile)
        copy_reports(result)
        (result / "timing.json").write_text(
            json.dumps({"test_runtime_s": round(runtime, 2)}, indent=2)
        )

        record["status"] = "PASS" if rc == 0 else "FAIL"
        record["runtime"] = runtime
        summary.append(record)
        print(f"{record['status']}  ({runtime:.1f}s)")

        if not args.keep_envs:
            shutil.rmtree(env)

    print_summary(summary)
    write_summary(summary)
    print(f"\nResults : {RESULT_DIR}")
    print(f"Summary : {COMPAT / 'summary.txt'}")

    if any(r["status"] not in ("PASS", "SKIP") for r in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
