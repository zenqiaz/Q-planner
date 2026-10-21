#!/usr/bin/env python3
"""Pin the behaviour of the OPI-backed status check in server_helpers.

Covers the swap made 2026-10-20: `status = "ok" if "ORCA TERMINATED NORMALLY" in out_text`
became `_terminated_normally(workdir, job_label)`, which runs OPI's grepper recipe on the file
`_read_out_text` would have read. The recipe greps the ``****``-delimited banner and is the
stricter of the two, so the regression that matters is a run changing class.

The wide check is `retrieval_diff.py` (2713 archived files, zero disagreements). This file pins
the edges that corpus does not reach: a missing .out, the glob fallback, and the empty string.

Runs standalone (`python test_retrieval_status.py`) or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server_helpers import (  # noqa: E402
    _read_out_text,
    _resolve_out_path,
    _terminated_normally,
)

HERE = Path(__file__).resolve().parent
SAMPLE_LIMIT = 200

OK_BANNER = "****ORCA TERMINATED NORMALLY****"
BAD_TEXT = "something went wrong\nno banner here\n"


def _legacy(out_text: str) -> bool:
    """The test this replaced."""
    return "ORCA TERMINATED NORMALLY" in out_text


# --------------------------------------------------------------------------
# Edges the archived corpus does not cover
# --------------------------------------------------------------------------

def test_missing_out_is_not_ok(tmpdir: Path | None = None):
    d = tmpdir or _mktmp("missing")
    assert _resolve_out_path(d, "job") is None
    assert _read_out_text(d, "job") == ""
    assert _terminated_normally(d, "job") is False, "no .out must read as error, as before"


def test_exact_label_preferred(tmpdir: Path | None = None):
    d = tmpdir or _mktmp("exact")
    (d / "job.out").write_text(OK_BANNER, encoding="utf-8")
    (d / "zzz_other.out").write_text(BAD_TEXT, encoding="utf-8")
    assert _resolve_out_path(d, "job").name == "job.out"
    assert _terminated_normally(d, "job") is True


def test_glob_fallback_when_label_absent(tmpdir: Path | None = None):
    """_read_out_text falls back to the last *.out; the status check must use the same file."""
    d = tmpdir or _mktmp("fallback")
    (d / "aaa.out").write_text(BAD_TEXT, encoding="utf-8")
    (d / "zzz.out").write_text(OK_BANNER, encoding="utf-8")
    assert _resolve_out_path(d, "nosuchlabel").name == "zzz.out"
    assert _terminated_normally(d, "nosuchlabel") is True
    assert _legacy(_read_out_text(d, "nosuchlabel")) is True


def test_empty_out_is_not_ok(tmpdir: Path | None = None):
    d = tmpdir or _mktmp("empty")
    (d / "job.out").write_text("", encoding="utf-8")
    assert _terminated_normally(d, "job") is False


def test_banner_without_asterisks_is_the_known_divergence(tmpdir: Path | None = None):
    """The one case where old and new genuinely differ.

    Not observed in 2713 archived ORCA 6.1.1 outputs, but it is the failure mode to watch if
    the banner format ever changes: the legacy test says ok, the recipe says not ok.
    """
    d = tmpdir or _mktmp("bare")
    (d / "job.out").write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
    assert _legacy(_read_out_text(d, "job")) is True
    assert _terminated_normally(d, "job") is False


# --------------------------------------------------------------------------
# Regression guard against real outputs, if any are present
# --------------------------------------------------------------------------

def test_agrees_with_legacy_on_archived_outputs():
    paths = []
    for p in HERE.rglob("*.out"):
        if p.is_file():
            paths.append(p)
        if len(paths) >= SAMPLE_LIMIT:
            break
    if not paths:
        print("  (no archived .out files found -- skipped)")
        return

    disagreements = []
    for p in paths:
        legacy = _legacy(p.read_text(encoding="utf-8", errors="ignore"))
        new = _terminated_normally(p.parent, p.stem)
        if legacy != new:
            disagreements.append((p, legacy, new))

    assert not disagreements, "\n".join(
        f"{p}: legacy={a} new={b}" for p, a, b in disagreements[:10]
    )
    print(f"  ({len(paths)} archived outputs, no disagreements)")


# --------------------------------------------------------------------------

def _mktmp(name: str) -> Path:
    import tempfile
    d = Path(tempfile.mkdtemp(prefix=f"retrstatus_{name}_"))
    return d


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
