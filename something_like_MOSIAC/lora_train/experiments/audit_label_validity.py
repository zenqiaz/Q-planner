"""Label-validity audit for the mined (functional, basis) pairs.

Motivation: the production filters validate the FUNCTIONAL vocabulary
(`ORCA_VALID_FUNCTIONALS`) and require the basis field to be non-null
(`generate_sft.is_clean`), but **nothing validates the basis field's contents**. Malformed bases
therefore pass every filter and become gold labels. Three have already been found by hand:

  * `B3LYP`/`b3lyp`  -- functional token in the basis field; 124 blank-upload metal rows in the
    sampled pool and 16 of 293 metal rows in the 2026-08-06 validation set
  * `PBE`/`pbepbe`   -- same defect, organic
  * basis `7=3)`     -- route-line parse artifact, organic

A selector can score exact-match points by reproducing such a string while a modal-pair constant
cannot, which inflates any margin measured against that constant.

This script classifies every record's basis field into one of:

  ok            recognisable basis-set name
  functional    the basis field holds a functional token
  placeholder   GEN / GENECP / CHKBASIS -- "custom basis defined elsewhere"
  artifact      contains characters a basis name cannot (=, parentheses imbalance, digits-only)
  unknown       not recognised; listed so the vocabulary can be extended rather than guessed at

CPU only, no DFT. Usage:  python audit_label_validity.py [--out report.json]
"""
from __future__ import annotations

import argparse, collections, json, re, sys
from pathlib import Path

ROOT_202606 = Path(r"D:\brick\D\202606\working\something_like MOSIAC")
ROOT_LORA = Path(r"D:\brick\D\20260217\working\something_like_MOSIAC")

# --- functional vocabulary, taken from the project's own tables ------------------------
FUNCTIONAL_TOKENS = {
    "B3LYP", "PBE", "PBE0", "PBE1PBE", "PBEPBE", "BLYP", "BP86", "B3PW91", "BEPBE",
    "TPSS", "TPSSH", "M06", "M06L", "M06-L", "M062X", "M06-2X", "M11L", "MN15L", "MN12L",
    "CAM-B3LYP", "WB97X", "WB97XD", "WB97X-D", "WB97X-D3", "WB97", "WB97M-V",
    "B2PLYP", "SVWN", "SVWN5", "SVWN3", "LDA", "HF", "RHF", "UHF",
    "MP2", "CCSD", "CCSDT", "CCSD(T)", "QCISD", "CISD", "CI", "CISDT", "FCI",
    "CASSCF", "CASPT2", "NEVPT2", "MRMP2", "MRCI", "DLPNO-CCSD(T)", "PM6", "PM7", "AM1", "PM3",
    "QCISD(T)", "LC-WPBE", "LC-WPBE-D3", "PBEH-3C", "B97-D3", "B3LYP-D3", "HSE06", "TPSSH-D3BJ",
}
PLACEHOLDERS = {"GEN", "GENECP", "CHKBASIS", "READ"}

# --- basis-name recognisers -----------------------------------------------------------
BASIS_PATTERNS = [
    r"^(ma-)?def2-?(SV|SVP|SVPD|TZVP|TZVPP|TZVPD|QZVP|QZVPP|TZV|mTZVPP)(-?J|/J|/C)?$",
    r"^(aug-)?cc-?pV[DTQ5-6]Z(-PP|-DK|-F12)?$",
    r"^(aug-)?cc-?pwCV[DTQ5]Z(-PP)?$",
    r"^6-31\+{0,2}G(\(\d*[dfp,+]*\)|\*{1,2})?$",
    r"^6-311\+{0,2}G(\([0-9dfpsg,+]*\)|\*{1,2})?$",
    r"^6-31\+{0,2}G(\([0-9dfpsg,+]*\))$",
    r"^CEP-(31|121)G\*?$", r"^def2-QZVPPD$", r"^def2-TZVP\(-f\)$", r"^def2-TZVPD$",
    r"^6-31G\(2df,p\)$", r"^6-311G\(3df,2p\)$", r"^6-311G\(2df,2pd\)$",
    r"^3-21G\*?$", r"^STO-?3G$", r"^4-31G$", r"^LANL2DZ$", r"^LANL08$",
    r"^SDD(All)?$", r"^SARC-?.*$", r"^ZORA-?def2-?.*$", r"^DKH-?def2-?.*$",
    r"^pc-?[0-4]$", r"^pcseg-?[0-4]$", r"^aug-pcseg-?[0-4]$",
    r"^(TZVP|SVP|QZVP|TZV|DZ|TZ|QZ|DZP|TZP)$",
    r"^EPR-?(II|III)$", r"^IGLO-?(II|III)$", r"^W1-?mtsmall$",
    r"^(aug-)?ANO-?.*$", r"^x2c-.*$", r"^old-?def2-?.*$", r"^Wachters\+f$",
]
BASIS_RX = [re.compile(p, re.I) for p in BASIS_PATTERNS]
ARTIFACT_RX = re.compile(r"[=;{}]|^\d+$|^\)|\($(?!.*\))")


def classify(basis) -> str:
    if basis is None or not str(basis).strip():
        return "null"
    b = str(basis).strip()
    u = b.upper().replace(" ", "")
    if u in PLACEHOLDERS:
        return "placeholder"
    if u in {t.upper() for t in FUNCTIONAL_TOKENS}:
        return "functional"
    if any(rx.match(b) for rx in BASIS_RX):
        return "ok"
    if ARTIFACT_RX.search(b):
        return "artifact"
    return "unknown"


def load_jsonl(p: Path):
    out = []
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def gold_from_sft(r):
    """Gold (functional, basis) from an SFT chat record's assistant turn."""
    for m in r.get("messages", []):
        if m.get("role") == "assistant":
            try:
                d = json.loads(m["content"])
                return d.get("functional"), d.get("basis")
            except Exception:
                return None, None
    return None, None


def report(name, pairs, out):
    """pairs: list of (functional, basis, cell)."""
    if not pairs:
        print(f"\n{name}: (absent)"); return
    cls = [(classify(b), f, b, c) for f, b, c in pairs]
    tot = len(cls)
    counts = collections.Counter(k for k, *_ in cls)
    bad = [x for x in cls if x[0] in ("functional", "artifact", "placeholder", "null")]
    print(f"\n{name}  n={tot:,}")
    for k in ("ok", "functional", "artifact", "placeholder", "null", "unknown"):
        if counts.get(k):
            print(f"    {k:12s} {counts[k]:7,}  {100*counts[k]/tot:6.2f}%")
    if bad:
        ex = collections.Counter((f, b) for _, f, b, _ in bad)
        print(f"    invalid total {len(bad):,} ({100*len(bad)/tot:.2f}%); "
              f"most common: {[(f'{f}/{b}', n) for (f, b), n in ex.most_common(4)]}")
        per_cell = collections.Counter(c for _, _, _, c in bad)
        if len(per_cell) > 1:
            print(f"    by cell: {dict(per_cell.most_common())}")
    unk = collections.Counter(b for k, _, b, _ in cls if k == "unknown")
    if unk:
        print(f"    unrecognised bases (extend the vocabulary rather than guess): "
              f"{[(b, n) for b, n in unk.most_common(6)]}")
    out[name] = {"n": tot, "counts": dict(counts),
                 "n_invalid": len(bad),
                 "pct_invalid": 100 * len(bad) / tot,
                 "examples": [{"functional": f, "basis": b, "n": n}
                              for (f, b), n in collections.Counter(
                                  (f, b) for _, f, b, _ in bad).most_common(10)],
                 "unrecognised": [{"basis": b, "n": n} for b, n in unk.most_common(20)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="label_validity_audit.json")
    args = ap.parse_args()
    out = {}

    print("=" * 94)
    print("LABEL-VALIDITY AUDIT -- (functional, basis) gold pairs")
    print("=" * 94)
    print("No production filter validates the basis field's contents; is_clean() only requires it")
    print("to be non-null, and ORCA_VALID_FUNCTIONALS constrains the functional only.")

    # stage 1: merged parse
    for nm, rel in (("merged parse", r"resampled_corpus\methods.jsonl"),
                    ("sampled pool", r"jsonl\methods.jsonl")):
        R = load_jsonl(ROOT_202606 / rel)
        report(f"{nm}  ({rel})",
               [(r.get("functional"), r.get("basis"), r.get("specialist_cell")) for r in R], out)

    # stage 2: the SFT splits Table 3 uses, and the molecule-split rebuild
    for d in ("jsonl", "jsonl_molecule_split"):
        pairs = []
        for cell in ("organic_general", "metal_general"):
            for kind in ("sft_%s.jsonl", "sft_val_%s.jsonl"):
                for r in load_jsonl(ROOT_LORA / "lora_train" / d / (kind % cell)):
                    f, b = gold_from_sft(r)
                    pairs.append((f, b, f"{cell}:{'val' if 'val' in kind else 'train'}"))
        report(f"SFT split  (lora_train/{d}/)", pairs, out)

    # stage 3: the August validation set actually scored
    aug = ROOT_LORA / "lora_train" / "experiments" / \
        "rag_sft_experiment_results_molecule_split_20260806T015251Z.json"
    if aug.exists():
        d = json.loads(aug.read_text(encoding="utf-8"))
        report("August 2026 validation (scored)",
               [(r.get("true_functional"), r.get("true_basis"), r.get("specialist_cell"))
                for r in d.get("val_results", [])], out)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
