"""
run_classical_baseline_molecule_split.py

Implements organic_general_classical_baseline_design.md (2026-08-06 update,
molecule-identity split as primary). Trains a classical, non-LLM multiclass
classifier (RandomForest + HistGradientBoostingClassifier -- sklearn's native
LightGBM-style histogram booster, used since neither lightgbm nor xgboost is
installed in this environment -- plus KNeighborsClassifier, the closest
classical analogue to RAG's retrieval mechanism, per user discussion) to
predict the exact (functional, basis) pair from structured features, on the
same molecule-identity split (jsonl_molecule_split/) used for the SFT/RAG
ablation (job 16038).

Feature set (per the design doc, matching organic_general_clustering_check.md /
insight #9's RF for comparability): log1p(n_atoms), charge, multiplicity!=1
flag, element-presence bits (fixed C/H/N/O/F for organic_general; all elements
observed in TRAIN for metal_general), task_type one-hot (categories observed
in TRAIN), solvent-presence flag, scale bucket (S<=10/M11-30/L31-100/XL>100)
one-hot. No SMILES features (coverage too sparse per the clustering check).

Rigor requirement (design doc "Rigor requirement" section): reuses the EXACT
same seeded train-sample entry_ids as SFT's job 16038
(rag_sft_experiment_inputs_molecule_split_20260806T004851Z.json's
train_sample_records) for the memorized-subset check, so classical vs. SFT
memorized/novel numbers are directly comparable at the same n per cell.

Usage:
    python run_classical_baseline_molecule_split.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAG_REPO = Path(r"D:\brick\D\20260217\working")
sys.path.insert(0, str(RAG_REPO))

import rag  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "jsonl_molecule_split"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ABLATION_INPUT_FILE = EXPERIMENTS_DIR / "rag_sft_experiment_inputs_molecule_split_20260806T004851Z.json"

PHASE1_CELLS = ["organic_general", "metal_general"]
ORGANIC_ELEMENTS = ["C", "H", "N", "O", "F"]
TASK_TYPES_FALLBACK = ["SP", "OPT", "FREQ", "TDDFT"]
SEED = 42


def scale_bucket(n_atoms) -> str:
    if not n_atoms:
        return "unknown"
    if n_atoms <= 10:
        return "S"
    if n_atoms <= 30:
        return "M"
    if n_atoms <= 100:
        return "L"
    return "XL"


def entry_ids_of(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(r["metadata"]["entry_id"])
    return out


class FeatureBuilder:
    """Fit on train records for a cell, then transform train/val/train-sample consistently."""

    def __init__(self, cell: str):
        self.cell = cell
        self.element_vocab: list[str] = []
        self.task_vocab: list[str] = []

    def fit(self, records: list[dict]) -> None:
        if self.cell == "organic_general":
            self.element_vocab = list(ORGANIC_ELEMENTS)
        else:
            elems = set()
            for r in records:
                elems.update(r.get("elements") or [])
            self.element_vocab = sorted(elems)

        tasks = sorted({r.get("task_type") or "UNKNOWN" for r in records})
        self.task_vocab = tasks

    def transform(self, records: list[dict]) -> np.ndarray:
        rows = []
        for r in records:
            n_atoms = r.get("n_atoms") or 0
            charge = r.get("charge") or 0
            mult = r.get("multiplicity") or 1
            elements = set(r.get("elements") or [])
            task = r.get("task_type") or "UNKNOWN"
            solvent = 1.0 if (r.get("solvent") or "").strip() else 0.0

            row = [np.log1p(max(n_atoms, 0)), float(charge), 1.0 if mult != 1 else 0.0, solvent]
            row += [1.0 if e in elements else 0.0 for e in self.element_vocab]
            row += [1.0 if task == t else 0.0 for t in self.task_vocab]
            bucket = scale_bucket(n_atoms)
            row += [1.0 if bucket == b else 0.0 for b in ["S", "M", "L", "XL", "unknown"]]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    def feature_names(self) -> list[str]:
        names = ["log1p_n_atoms", "charge", "mult_ne_1", "has_solvent"]
        names += [f"elem_{e}" for e in self.element_vocab]
        names += [f"task_{t}" for t in self.task_vocab]
        names += [f"scale_{b}" for b in ["S", "M", "L", "XL", "unknown"]]
        return names


def joint_label(r: dict) -> str:
    return f"{r.get('functional')}\x1f{r.get('basis')}"


def score(records: list[dict], pred_functional: list, pred_basis: list) -> dict:
    n = len(records)
    func_match = sum(1 for r, pf in zip(records, pred_functional) if pf == r.get("functional"))
    basis_match = sum(1 for r, pb in zip(records, pred_basis) if pb == r.get("basis"))
    both_match = sum(
        1 for r, pf, pb in zip(records, pred_functional, pred_basis)
        if pf == r.get("functional") and pb == r.get("basis")
    )
    return {
        "n": n,
        "functional_match_pct": round(100 * func_match / n, 1) if n else 0.0,
        "basis_match_pct": round(100 * basis_match / n, 1) if n else 0.0,
        "both_match_pct": round(100 * both_match / n, 1) if n else 0.0,
    }


def predict_functional_basis(model, le: LabelEncoder, X) -> tuple[list, list]:
    pred_idx = model.predict(X)
    pred_joint = le.inverse_transform(pred_idx)
    pred_functional, pred_basis = [], []
    for j in pred_joint:
        f, b = j.split("\x1f", 1)
        pred_functional.append(f)
        pred_basis.append(b)
    return pred_functional, pred_basis


def main():
    print("[classical baseline] loading raw, unfiltered pool for entry_id lookup...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    by_id = {r["entry_id"]: r for r in raw_pool if r.get("entry_id")}
    print(f"[classical baseline] raw pool: {len(raw_pool)} records, {len(by_id)} unique entry_ids",
          file=sys.stderr)

    ablation_payload = json.loads(ABLATION_INPUT_FILE.read_text(encoding="utf-8"))
    train_sample_ids_by_cell: dict[str, list[str]] = {}
    for rec in ablation_payload.get("train_sample_records", []):
        train_sample_ids_by_cell.setdefault(rec["specialist_cell"], []).append(rec["entry_id"])
    print(f"[classical baseline] loaded train-sample entry_ids from {ABLATION_INPUT_FILE.name} "
          f"(reusing job 16038's exact seeded sample for the memorized-subset check)", file=sys.stderr)

    all_results = {}
    t0 = time.monotonic()

    for cell in PHASE1_CELLS:
        train_ids = entry_ids_of(SPLIT_DIR / f"sft_{cell}.jsonl")
        val_ids = entry_ids_of(SPLIT_DIR / f"sft_val_{cell}.jsonl")
        missing = [i for i in train_ids + val_ids if i not in by_id]
        if missing:
            raise RuntimeError(f"{cell}: {len(missing)} entry_ids not found in raw pool")

        train_records = [dict(by_id[i], specialist_cell=cell) for i in train_ids]
        val_records = [dict(by_id[i], specialist_cell=cell) for i in val_ids]
        mem_ids = train_sample_ids_by_cell.get(cell, [])
        train_by_id = {r["entry_id"]: r for r in train_records}
        mem_records = [train_by_id[i] for i in mem_ids if i in train_by_id]
        print(f"\n[classical baseline] {cell}: train={len(train_records)} val={len(val_records)} "
              f"memorized-sample={len(mem_records)}", file=sys.stderr)

        fb = FeatureBuilder(cell)
        fb.fit(train_records)
        X_train = fb.transform(train_records)
        X_val = fb.transform(val_records)
        X_mem = fb.transform(mem_records)

        y_train_joint = [joint_label(r) for r in train_records]
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_joint)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_mem_s = scaler.transform(X_mem) if len(mem_records) else X_mem

        cell_results = {"n_features": X_train.shape[1], "feature_names": fb.feature_names()}

        # --- RandomForest, light grid search selected by val both_match ---
        rf_grid = [
            {"n_estimators": 200, "max_depth": None},
            {"n_estimators": 500, "max_depth": None},
            {"n_estimators": 200, "max_depth": 20},
            {"n_estimators": 500, "max_depth": 20},
        ]
        best_rf = None
        best_rf_score = -1.0
        best_rf_params = None
        for params in rf_grid:
            m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
            m.fit(X_train, y_train)
            pf, pb = predict_functional_basis(m, le, X_val)
            s = score(val_records, pf, pb)
            if s["both_match_pct"] > best_rf_score:
                best_rf_score, best_rf, best_rf_params = s["both_match_pct"], m, params
        print(f"  RandomForest best params (by val both_match): {best_rf_params} -> {best_rf_score}%",
              file=sys.stderr)

        # --- HistGradientBoostingClassifier (LightGBM/XGBoost substitute -- neither installed) ---
        hgb_grid = [
            {"max_iter": 200, "max_depth": None},
            {"max_iter": 500, "max_depth": None},
            {"max_iter": 200, "max_depth": 10},
            {"max_iter": 500, "max_depth": 10},
        ]
        best_hgb = None
        best_hgb_score = -1.0
        best_hgb_params = None
        for params in hgb_grid:
            m = HistGradientBoostingClassifier(random_state=SEED, **params)
            m.fit(X_train, y_train)
            pf, pb = predict_functional_basis(m, le, X_val)
            s = score(val_records, pf, pb)
            if s["both_match_pct"] > best_hgb_score:
                best_hgb_score, best_hgb, best_hgb_params = s["both_match_pct"], m, params
        print(f"  HistGradientBoosting best params (by val both_match): {best_hgb_params} -> {best_hgb_score}%",
              file=sys.stderr)

        # --- k-NN (closest classical analogue to RAG's retrieval mechanism) ---
        knn_grid = [{"n_neighbors": 1}, {"n_neighbors": 5}, {"n_neighbors": 10}]
        best_knn = None
        best_knn_score = -1.0
        best_knn_params = None
        for params in knn_grid:
            m = KNeighborsClassifier(**params)
            m.fit(X_train_s, y_train)
            pf, pb = predict_functional_basis(m, le, X_val_s)
            s = score(val_records, pf, pb)
            if s["both_match_pct"] > best_knn_score:
                best_knn_score, best_knn, best_knn_params = s["both_match_pct"], m, params
        print(f"  k-NN best params (by val both_match): {best_knn_params} -> {best_knn_score}%",
              file=sys.stderr)

        models = {
            "random_forest": (best_rf, best_rf_params, X_train, X_val, X_mem),
            "hist_gradient_boosting": (best_hgb, best_hgb_params, X_train, X_val, X_mem),
            "knn": (best_knn, best_knn_params, X_train_s, X_val_s, X_mem_s),
        }

        cell_results["models"] = {}
        for name, (model, params, Xtr, Xv, Xm) in models.items():
            pf_val, pb_val = predict_functional_basis(model, le, Xv)
            novel = score(val_records, pf_val, pb_val)
            entry = {"best_params": params, "novel_subset": novel}
            if len(mem_records):
                pf_mem, pb_mem = predict_functional_basis(model, le, Xm)
                memorized = score(mem_records, pf_mem, pb_mem)
                entry["memorized_subset"] = memorized
                entry["memorization_gap_pp"] = round(
                    memorized["both_match_pct"] - novel["both_match_pct"], 1)
            cell_results["models"][name] = entry

        all_results[cell] = cell_results

    duration_s = time.monotonic() - t0

    print(f"\n{'=' * 100}\nSUMMARY (duration {duration_s:.1f}s)\n{'=' * 100}")
    print(f"{'cell':<18} {'model':<24} {'n(val)':>7} {'func%':>7} {'basis%':>7} {'both%':>7}   "
          f"{'memorized both%':>16} {'gap(pp)':>8}")
    for cell, cres in all_results.items():
        for name, entry in cres["models"].items():
            nov = entry["novel_subset"]
            mem = entry.get("memorized_subset")
            mem_str = f"{mem['both_match_pct']:.1f}%" if mem else "n/a"
            gap_str = f"{entry.get('memorization_gap_pp', 'n/a')}" if mem else "n/a"
            print(f"{cell:<18} {name:<24} {nov['n']:>7} {nov['functional_match_pct']:>6.1f}% "
                  f"{nov['basis_match_pct']:>6.1f}% {nov['both_match_pct']:>6.1f}%   "
                  f"{mem_str:>16} {gap_str:>8}")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = EXPERIMENTS_DIR / f"classical_baseline_results_molecule_split_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": "classical_baseline_results_molecule_split",
            "timestamp_utc": ts,
            "design_doc": "organic_general_classical_baseline_design.md",
            "split": "jsonl_molecule_split (molecule-identity, primary per 2026-08-06 update)",
            "note": "hist_gradient_boosting substitutes for LightGBM/XGBoost -- neither installed "
                    "in this environment; knn added per user discussion as the closest classical "
                    "analogue to RAG's retrieval mechanism (not in the original design doc's "
                    "headline models, kept as a comparison row).",
            "duration_s": duration_s,
            "results": all_results,
        }, f, indent=2, ensure_ascii=True)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
