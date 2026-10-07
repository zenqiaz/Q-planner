"""
boundary_analysis.py

Tests whether a linear decision boundary (a "hyperplane") in molecular-feature space separates
"setting A wins" from "setting B wins", given real per-reaction head-to-head accuracy data for
two (functional, basis) settings. Built 2026-09-08 for the paper-discussion boundary experiment:
we cannot train real Voronoi-cell routers (too few data entries per cell), but we can test
whether the underlying decision surface the paper wants to route on is real and low-complexity,
rather than indistinguishable from noise -- a low-capacity linear model + leave-one-out CV +
permutation test, matched to the very small N (~35-234) this project actually has.

Extended 2026-09-11 per discussion-draft follow-up: this file makes an EXISTENCE claim only
("a boundary is real"), never a CONSTRUCTION claim ("here is the deployable router") -- see
`decision_boundary_experiment_design.md`. Two additions support that scope directly:

1. --model {logreg,svm}: a linear SVM (LinearSVC) alongside logistic regression. Both are
   linear/low-capacity and give near-identical answers here; offered so the reported result
   isn't tied to one arbitrary classifier choice.

2. A k-means "anchor" check (--kmeans-k, on by default, k=2): fit k-means on the SAME
   (optionally --features-restricted) standardized feature space, BLIND to the win/lose label,
   and compare it two ways:
     (a) k-means cluster membership vs. the TRUE win/lose label directly (best-alignment
         accuracy + a label-permutation p-value) -- a classifier-free existence test: even an
         unsupervised partition of feature space predicts which setting wins, better than
         chance. This deliberately avoids ever writing down a fitted decision function.
     (b) the fitted linear classifier's boundary normal vs. the k-means centroid-difference
         direction (cosine similarity + a label-permutation p-value) -- a sanity check that the
         SUPERVISED fit isn't a skewed/overfit artifact of small-N label noise: if the fitted
         boundary is anchored to the natural, label-blind geometry of the data, its direction
         should align with where k-means would split the same points anyway.
   Both checks are only meaningful in a low-dimensional, chemically-motivated feature space
   (see --features) -- in the full ~8-feature vector the k-means split is usually dominated by
   whichever feature has the most spread (molecule size), not the hypothesized chemical axis.

Two ways to get input data (both produce the same per-reaction abs-error-for-two-settings shape):
  --pair-log PATH        run_specialist_pair_experiment.py's own output (pair_a/pair_b already
                          picked to be a genuine chemical trade-off, real new ORCA compute).
  --accuracy-log PATH --cell CELL --cond-a NAME --cond-b NAME
                          reuse two of run_accuracy_benchmark.py's own 7 conditions (e.g.
                          fixed_default vs skills_heuristic) -- free, already-computed, but both
                          are general-purpose settings so the expected effect size is smaller.

Usage:
    python boundary_analysis.py --accuracy-log accuracy_benchmark_logs/accuracy_benchmark_20260908T043300Z.json \\
        --cell organic_general --cond-a fixed_default --cond-b skills_heuristic \\
        --n-permutations 5000 --plot-out boundary_fixed_vs_skills.png

    python boundary_analysis.py --pair-log accuracy_benchmark_logs/specialist_pair_organic_general_..._....json \\
        --model svm --features degree_unsaturation,n_atoms \\
        --n-permutations 5000 --plot-out boundary_pbe0_vs_qcisd.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import run_accuracy_benchmark as arb  # noqa: E402 -- reuse load_ground_truth/reaction_profile

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402


def _element_counts(xyz_angstrom: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in xyz_angstrom.strip().splitlines():
        el = line.split()[0]
        counts[el] = counts.get(el, 0) + 1
    return counts


# Optional per-species computed descriptors, keyed by species_id, loaded from
# --species-descriptors. Added 2026-09-17: metal_general's boundary test came back null with the
# element-one-hot placeholder (design doc Open Question 2), so this is the channel for a REAL
# electronic-structure descriptor -- e.g. the frontier gap extracted from the specialist-pair
# ORCA outputs, a direct computed proxy for ligand-field splitting, at zero new compute.
# CAVEAT to state wherever this is used: such a descriptor is POST-HOC -- obtaining it requires
# running the calculation, so it can characterise a boundary but cannot route to one in advance.
# That is acceptable for this project's existence-only claim, and must not be quietly upgraded
# into a routing claim.
SPECIES_DESCRIPTORS: dict[str, dict[str, float]] = {}


def _representative_species_id(reaction: dict, species: dict[str, dict]) -> str | None:
    rep = arb.representative_species(reaction, species)
    for sid in reaction.get("species_ids", []):
        if species.get(sid) is rep:
            return sid
    return None


def build_feature_row(reaction: dict, species: dict[str, dict], cell: str) -> dict[str, float]:
    """Same representative-species convention as this project's own RAG/RF selectors
    (reaction_profile()) -- the boundary test uses the identical feature basis the paper's other
    selectors already use, so it's a fair apples-to-apples comparison, not a new feature scheme
    invented just for this experiment."""
    profile = arb.reaction_profile(reaction, species, cell)
    rep = arb.representative_species(reaction, species)
    counts = _element_counts(rep["xyz_angstrom"])
    row: dict[str, float] = {
        "n_atoms": float(profile["n_atoms"]),
        "charge": float(profile["charge"]),
        "multiplicity": float(profile["multiplicity"]),
    }
    if cell == "organic_general":
        nC, nH, nN, nO, nF = (counts.get(e, 0) for e in ("C", "H", "N", "O", "F"))
        row["n_C"] = float(nC)
        row["n_N"] = float(nN)
        row["n_O"] = float(nO)
        row["n_F"] = float(nF)
        # Degree of unsaturation -- ties directly to insight #15 mechanism 1 (rare-precedent
        # extrapolation, triple-bonded systems), the chemically motivated candidate boundary axis.
        row["degree_unsaturation"] = (2 * nC + 2 + nN - nH - nF) / 2.0
    else:
        # metal_general: one-hot the specific TM/heavy element present (small, known vocabulary)
        # -- NOT a ligand-field-strength feature (flagged as not yet implemented; would need a
        # per-ligand spectrochemical-series classification this project doesn't have on hand).
        for el in profile["elements"]:
            if el not in ("C", "H", "N", "O", "F"):
                row[f"has_{el}"] = 1.0

    if SPECIES_DESCRIPTORS:
        sid = _representative_species_id(reaction, species)
        for k, v in (SPECIES_DESCRIPTORS.get(sid) or {}).items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
    return row


def load_from_accuracy_log(
    path: Path, cell: str, cond_a: str, cond_b: str
) -> tuple[list[dict], list[str], list[int], list[float], list[float]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    pr = d["cells"][cell]["scoring"]["per_reaction"]
    gt = arb.load_ground_truth(cell, None, arb.random.Random(0))
    species, reactions_by_id = gt["species"], {r["reaction_id"]: r for r in gt["reactions"]}
    rows, ids, labels, errs_a, errs_b = [], [], [], [], []
    for r in pr:
        c = r.get("conditions", {})
        a, b = c.get(cond_a, {}), c.get(cond_b, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        ea, eb = a["abs_error_kcal_mol"], b["abs_error_kcal_mol"]
        if ea == eb:
            continue  # exact tie -- no boundary to assign it to
        reaction = reactions_by_id.get(r["reaction_id"])
        if reaction is None:
            continue
        rows.append(build_feature_row(reaction, species, cell))
        ids.append(r["reaction_id"])
        labels.append(1 if ea < eb else 0)
        errs_a.append(float(ea))
        errs_b.append(float(eb))
    return rows, ids, labels, errs_a, errs_b


def load_from_pair_log(path: Path) -> tuple[list[dict], list[str], list[int], list[float], list[float], str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    cell = d["cell"]
    gt = arb.load_ground_truth(cell, None, arb.random.Random(0))
    species, reactions_by_id = gt["species"], {r["reaction_id"]: r for r in gt["reactions"]}
    rows, ids, labels, errs_a, errs_b = [], [], [], [], []
    for r in d["per_reaction"]:
        ea, eb = r["pair_a_abs_error_kcal_mol"], r["pair_b_abs_error_kcal_mol"]
        if ea == eb:
            continue
        reaction = reactions_by_id.get(r["reaction_id"])
        if reaction is None:
            continue
        rows.append(build_feature_row(reaction, species, cell))
        ids.append(r["reaction_id"])
        labels.append(1 if ea < eb else 0)
        errs_a.append(float(ea))
        errs_b.append(float(eb))
    return rows, ids, labels, errs_a, errs_b, cell


def to_matrix(rows: list[dict], features: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    cols = sorted({k for row in rows for k in row})
    if features is not None:
        missing = [f for f in features if f not in cols]
        if missing:
            raise SystemExit(f"--features named columns not present in the built feature set: {missing} "
                              f"(available: {cols})")
        cols = features
    X = np.array([[row.get(c, 0.0) for c in cols] for row in rows])
    return X, cols


def make_classifier(model: str):
    if model == "logreg":
        return LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    if model == "svm":
        # dual=False: primal formulation, correct choice whenever n_samples > n_features (always
        # true here, N~35-234 vs <=8 features) and avoids sklearn's version-dependent `dual`
        # default warning.
        return LinearSVC(C=1.0, max_iter=5000, dual=False)
    raise SystemExit(f"unknown --model {model!r} (choices: logreg, svm)")


def loocv_accuracy(X: np.ndarray, y: np.ndarray, model: str = "logreg") -> tuple[float, float]:
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        clf = make_classifier(model)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        preds[test_idx] = clf.predict(scaler.transform(X[test_idx]))
    acc = float(np.mean(preds == y))
    bal_acc = balanced_accuracy_score(y, preds)
    return acc, bal_acc


def _best_alignment_accuracy(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Agreement between two binary partitions, taking whichever of the two possible 0/1
    mappings maximizes agreement -- cluster/class identity from k-means (and from a classifier's
    arbitrary positive-class choice) is not meaningfully ordered, so a raw (labels_a==labels_b)
    comparison would be an artifact of which side happened to get called 0 vs 1."""
    agree = float(np.mean(labels_a == labels_b))
    return max(agree, 1.0 - agree)


def kmeans_anchor_stats(
    Xs: np.ndarray, y: np.ndarray, clf_coef: np.ndarray, km_labels: np.ndarray, centroid_diff: np.ndarray,
    clf_preds: np.ndarray,
) -> dict[str, float]:
    """The two k-means-as-anchor numbers described in the module docstring, for one fit
    (real data or one permutation draw). `km_labels`/`centroid_diff` are fixed (computed once
    from the unpermuted, label-blind k-means fit) and passed in unchanged across permutations;
    only `y`/`clf_coef`/`clf_preds` vary per permutation draw."""
    cos_to_anchor = float(abs(
        np.dot(clf_coef, centroid_diff) / (np.linalg.norm(clf_coef) * np.linalg.norm(centroid_diff) + 1e-12)
    ))
    km_vs_label = _best_alignment_accuracy(km_labels, y)
    km_vs_boundary = _best_alignment_accuracy(km_labels, clf_preds)
    return {
        "cos_to_anchor": cos_to_anchor,
        "km_vs_label_agreement": km_vs_label,
        "km_vs_boundary_agreement": km_vs_boundary,
    }


def permutation_test(
    X: np.ndarray, y: np.ndarray, n_permutations: int, model: str = "logreg", seed: int = 0,
    km_labels: np.ndarray | None = None, centroid_diff: np.ndarray | None = None,
) -> dict[str, Any]:
    """Runs the existence-proof LOOCV-accuracy permutation test (unchanged from the original
    version of this file), and -- when a k-means anchor is supplied -- extends the SAME
    permutation loop to also build null distributions for the two anchor statistics above, at
    the marginal cost of one extra full-data fit per permutation (the LOOCV fits are already
    being done regardless)."""
    real_acc, _ = loocv_accuracy(X, y, model)

    scaler_full = StandardScaler().fit(X)
    Xs_full = scaler_full.transform(X)
    clf_full = make_classifier(model).fit(Xs_full, y)
    clf_coef_full = clf_full.coef_[0]
    clf_preds_full = clf_full.predict(Xs_full)

    anchor_available = km_labels is not None and centroid_diff is not None
    real_anchor = (
        kmeans_anchor_stats(Xs_full, y, clf_coef_full, km_labels, centroid_diff, clf_preds_full)
        if anchor_available else None
    )

    rng = np.random.default_rng(seed)
    null_accs = np.empty(n_permutations)
    null_cos = np.empty(n_permutations) if anchor_available else None
    null_km_vs_label = np.empty(n_permutations) if anchor_available else None

    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        null_accs[i], _ = loocv_accuracy(X, y_shuffled, model)
        if anchor_available:
            clf_i = make_classifier(model).fit(Xs_full, y_shuffled)
            stats_i = kmeans_anchor_stats(
                Xs_full, y_shuffled, clf_i.coef_[0], km_labels, centroid_diff, clf_i.predict(Xs_full)
            )
            null_cos[i] = stats_i["cos_to_anchor"]
            null_km_vs_label[i] = stats_i["km_vs_label_agreement"]

    p_acc = (1 + np.sum(null_accs >= real_acc)) / (n_permutations + 1)
    result: dict[str, Any] = {
        "real_acc": real_acc, "p_acc": p_acc, "null_accs": null_accs,
        "clf_coef_full": clf_coef_full, "clf_preds_full": clf_preds_full,
    }
    if anchor_available:
        p_cos = (1 + np.sum(null_cos >= real_anchor["cos_to_anchor"])) / (n_permutations + 1)
        p_km_label = (1 + np.sum(null_km_vs_label >= real_anchor["km_vs_label_agreement"])) / (n_permutations + 1)
        result.update({
            "real_anchor": real_anchor, "p_cos": p_cos, "null_cos": null_cos,
            "p_km_label": p_km_label, "null_km_vs_label": null_km_vs_label,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accuracy-log", type=Path, default=None)
    parser.add_argument("--cell", default=None, choices=["organic_general", "metal_general"])
    parser.add_argument("--cond-a", default=None)
    parser.add_argument("--cond-b", default=None)
    parser.add_argument("--pair-log", type=Path, default=None)
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm"],
                         help="Linear classifier for the existence-proof fit. Both are low-capacity "
                              "and expected to agree closely; not tied to one arbitrary choice.")
    parser.add_argument("--features", default=None,
                         help="Comma-separated subset of feature-column names to restrict BOTH the "
                              "classifier and the k-means anchor to (e.g. 'degree_unsaturation,n_atoms'). "
                              "Strongly recommended over the full feature vector when the k-means anchor "
                              "is enabled -- see module docstring.")
    parser.add_argument("--kmeans-k", type=int, default=2,
                         help="k for the label-blind k-means anchor. 2 matches the binary win/lose "
                              "label and is the only value the cosine-to-anchor check is defined for.")
    parser.add_argument("--species-descriptors", type=Path, default=None,
                         help="JSON {species_id: {name: value}} of precomputed per-species "
                              "descriptors (e.g. frontier gaps) merged into the feature row for "
                              "each reaction's representative species. Post-hoc by nature -- see "
                              "SPECIES_DESCRIPTORS' comment.")
    parser.add_argument("--no-kmeans-anchor", action="store_true",
                         help="Skip the k-means anchor entirely and only report the original "
                              "LOOCV-accuracy existence test.")
    parser.add_argument("--n-permutations", type=int, default=2000,
                         help="Lower than a publication-grade 10,000 by default so a first pilot "
                              "run finishes in seconds -- raise this for the real reported number.")
    parser.add_argument("--plot-out", type=Path, default=None)
    args = parser.parse_args()

    if args.species_descriptors:
        global SPECIES_DESCRIPTORS
        SPECIES_DESCRIPTORS = json.loads(args.species_descriptors.read_text(encoding="utf-8"))
        print(f"loaded per-species descriptors for {len(SPECIES_DESCRIPTORS)} species "
              f"from {args.species_descriptors}")

    if args.pair_log:
        rows, ids, labels, errs_a, errs_b, cell = load_from_pair_log(args.pair_log)
        label_a, label_b = "pair_a", "pair_b"
    else:
        if not (args.accuracy_log and args.cell and args.cond_a and args.cond_b):
            raise SystemExit("Either --pair-log, or --accuracy-log/--cell/--cond-a/--cond-b, is required.")
        rows, ids, labels, errs_a, errs_b = load_from_accuracy_log(
            args.accuracy_log, args.cell, args.cond_a, args.cond_b
        )
        cell = args.cell
        label_a, label_b = args.cond_a, args.cond_b

    y = np.array(labels)
    errs_a_arr, errs_b_arr = np.array(errs_a), np.array(errs_b)
    features = args.features.split(",") if args.features else None
    X, cols = to_matrix(rows, features)
    n_a, n_b = int(np.sum(y == 1)), int(np.sum(y == 0))
    print(f"cell={cell}  n={len(y)}  {label_a} wins={n_a}  {label_b} wins={n_b}  "
          f"majority-class floor={max(n_a, n_b)/len(y):.3f}")
    print(f"model={args.model}  features ({len(cols)}): {cols}")

    km_labels = centroid_diff = None
    if not args.no_kmeans_anchor:
        scaler_km = StandardScaler().fit(X)
        km = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=0).fit(scaler_km.transform(X))
        km_labels = km.labels_
        if args.kmeans_k == 2:
            centroid_diff = km.cluster_centers_[1] - km.cluster_centers_[0]
        km_sizes = [int(np.sum(km_labels == c)) for c in range(args.kmeans_k)]
        print(f"k-means anchor (k={args.kmeans_k}, blind to label): cluster sizes = {'/'.join(map(str, km_sizes))}")
        if args.kmeans_k != 2:
            print("  (cosine-to-anchor and balance checks below are only defined for k=2; skipped)")

    result = permutation_test(X, y, args.n_permutations, args.model, km_labels=km_labels, centroid_diff=centroid_diff)

    print(f"LOOCV accuracy = {result['real_acc']:.3f}")
    print(f"null distribution: mean={result['null_accs'].mean():.3f} std={result['null_accs'].std():.3f} "
          f"(n_permutations={args.n_permutations})")
    print(f"permutation p-value = {result['p_acc']:.4f}  "
          f"({'SIGNIFICANT at p<0.05 -- a real boundary, not noise' if result['p_acc'] < 0.05 else 'not significant at p<0.05 given this N/feature set'})")

    if "real_anchor" in result:
        ra = result["real_anchor"]
        print(f"\nk-means-vs-true-label agreement (classifier-free existence check) = "
              f"{ra['km_vs_label_agreement']:.3f}  (majority-class floor = {max(n_a, n_b)/len(y):.3f})")
        print(f"  permutation p-value = {result['p_km_label']:.4f}  "
              f"({'SIGNIFICANT' if result['p_km_label'] < 0.05 else 'not significant'} at p<0.05 -- "
              f"even a label-blind partition of feature space predicts the winner)")
        print(f"cosine(classifier boundary, k-means anchor direction) = {ra['cos_to_anchor']:.3f}  "
              f"(1.0 = boundary exactly parallel to the natural label-blind cluster split)")
        print(f"  permutation p-value = {result['p_cos']:.4f}  "
              f"({'SIGNIFICANT' if result['p_cos'] < 0.05 else 'not significant'} at p<0.05 -- the fitted "
              f"boundary's direction is anchored to real feature-space geometry, not small-N label noise)")
        print(f"classifier-vs-k-means partition agreement = {ra['km_vs_boundary_agreement']:.3f}  "
              f"(balance sanity check: classifier split {int(np.sum(result['clf_preds_full']==0))}/"
              f"{int(np.sum(result['clf_preds_full']==1))} vs k-means split {'/'.join(map(str, km_sizes))})")

        print("\nReal-execution MAE by k-means cluster (not just statistical significance -- "
              "the practical stakes on each side):")
        for c in sorted(set(km_labels)):
            mask = km_labels == c
            mae_a, mae_b = float(errs_a_arr[mask].mean()), float(errs_b_arr[mask].mean())
            winner = label_a if mae_a < mae_b else label_b
            print(f"    cluster {c} (n={int(mask.sum())}): {label_a} MAE={mae_a:.2f}  "
                  f"{label_b} MAE={mae_b:.2f}  -> {winner} wins this cluster")

    coefs = sorted(zip(cols, result["clf_coef_full"]), key=lambda kv: -abs(kv[1]))
    print("\nStandardized coefficients (full-data fit, for interpretation only -- LOOCV/permutation "
          "numbers above are the actual evidence):")
    for name, c in coefs:
        print(f"    {name:20s} {c:+.3f}  (positive -> favors {label_a})")

    if args.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top2 = [c for c, _ in coefs[:2]] if len(cols) > 2 else cols[:2]
        i0, i1 = cols.index(top2[0]), cols.index(top2[1])
        X2 = X[:, [i0, i1]]
        scaler2 = StandardScaler().fit(X2)
        clf2 = make_classifier(args.model).fit(scaler2.transform(X2), y)
        acc2, _ = loocv_accuracy(X2, y, args.model)

        rng_jitter = np.random.default_rng(1)
        jitter = lambda v: v + rng_jitter.uniform(-0.12, 0.12, size=v.shape)  # noqa: E731 -- discrete integer features overplot badly otherwise

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for cls, marker, color, name in ((1, "o", "#1A375E", label_a), (0, "^", "#B5651D", label_b)):
            mask = y == cls
            ax.scatter(jitter(X2[mask, 0]), jitter(X2[mask, 1]), marker=marker, color=color,
                       label=f"{name} wins", edgecolor="white", s=60, alpha=0.85)

        xs = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 200)
        ys = np.linspace(X2[:, 1].min(), X2[:, 1].max(), 200)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        zz = clf2.predict(scaler2.transform(grid)).reshape(gx.shape)
        ax.contour(gx, gy, zz, levels=[0.5], colors="black", linewidths=1.5)

        if not args.no_kmeans_anchor:
            # Illustrative-only: a SEPARATE k=2 fit restricted to these same 2 plotted features, so
            # the dashed anchor line lives in the same 2D subspace as the plot -- distinct from the
            # full-feature-space anchor check reported above, which this is not a substitute for.
            km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(scaler2.transform(X2))
            zz_km = km2.predict(scaler2.transform(grid)).reshape(gx.shape)
            ax.contour(gx, gy, zz_km, levels=[0.5], colors="#888888", linewidths=1.5, linestyles="dashed")
            ax.plot([], [], color="#888888", linestyle="dashed", label="k-means split (illustrative, 2-feature)")

        ax.set_xlabel(top2[0] + " (jittered for display)")
        ax.set_ylabel(top2[1] + " (jittered for display)")
        ax.set_title(f"{cell}: {label_a} vs {label_b}\n"
                     f"illustrative 2-feature boundary (2-feature LOOCV acc = {acc2:.2f})",
                     fontsize=11)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=150)
        print(f"\nWrote plot to {args.plot_out} (illustrative 2-feature fit, not the full-feature "
              f"statistical result reported above)")


if __name__ == "__main__":
    main()
