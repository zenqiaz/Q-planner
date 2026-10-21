#!/usr/bin/env python3
"""Pin the INPUT_ERROR classification added 2026-10-20.

Output goes to stderr throughout: importing `server_with_product` closes stdout as a side
effect of its MCP stdio setup, so a plain print after the import raises
``ValueError: I/O operation on closed file``.

Runs standalone (`python test_error_codes.py`) or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server_with_product import _classify_orca_error_code  # noqa: E402
from build_graph_from_plan import _infer_error_code  # noqa: E402

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


# The banner exactly as ORCA 6.1.1 prints it.
INPUT_ERR = """
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                 INPUT ERROR
            UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE
              GEN
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

NORMAL = "SCF CONVERGED AFTER 9 CYCLES\n****ORCA TERMINATED NORMALLY****\n"


def test_input_error_is_classified():
    assert _classify_orca_error_code(INPUT_ERR) == "INPUT_ERROR"


def test_normal_output_has_no_code():
    assert _classify_orca_error_code(NORMAL) is None


def test_input_error_outranks_scf_text():
    """Precedence is the point: input that never parsed must not be retried as an SCF problem."""
    both = INPUT_ERR + "\nSCF NOT CONVERGED AFTER 125 CYCLES\n"
    assert _classify_orca_error_code(both) == "INPUT_ERROR"


def test_existing_codes_still_work():
    assert _classify_orca_error_code("SCF NOT CONVERGED AFTER 125 CYCLES") == "SCF_NOT_CONVERGED"
    assert _classify_orca_error_code("***IMAGINARY MODE*** at -230 cm-1") == "IMAG_FREQ"
    assert _classify_orca_error_code("THE CAS PROCEDURE HAS NOT CONVERGED") == "CASSCF_NOT_CONVERGED"


def test_client_passes_server_code_through_unchanged():
    assert _infer_error_code({"code": "INPUT_ERROR"}) == "INPUT_ERROR"


def test_client_fallback_matches_server_precedence():
    """When no server code is present, the tail heuristic must rank the two the same way."""
    assert _infer_error_code({"tail": INPUT_ERR}) == "INPUT_ERROR"
    both = INPUT_ERR + "\nSCF NOT CONVERGED\n"
    assert _infer_error_code({"tail": both}) == "INPUT_ERROR"


def test_corpus_counts_match_the_measurement():
    """79 of 191 failures, 0 of 2522 successes -- the numbers the change was justified on."""
    import csv
    csv_path = HERE / "retrieval_diff_rows.csv"
    if not csv_path.exists():
        log("  (retrieval_diff_rows.csv absent -- run retrieval_diff.py first; skipped)")
        return

    hits_fail = hits_ok = n_fail = n_ok = 0
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = Path(row["path"])
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hit = _classify_orca_error_code(text) == "INPUT_ERROR"
            if row["cur_ok"] == "True":
                n_ok += 1
                hits_ok += hit
            else:
                n_fail += 1
                hits_fail += hit

    log(f"  ({n_ok} successes, {hits_ok} INPUT_ERROR; {n_fail} failures, {hits_fail} INPUT_ERROR)")
    assert hits_ok == 0, f"INPUT_ERROR fired on {hits_ok} successful runs"
    assert hits_fail > 0, "INPUT_ERROR fired on no failure -- pattern is dead"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            log(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            log(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    log("")
    log(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
