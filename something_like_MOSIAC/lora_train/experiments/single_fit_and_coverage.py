import io, json, collections

RES = r"D:\brick\D\20260217\working\something_like_MOSIAC\lora_train\experiments\rag_sft_experiment_results_20260803T075244Z.json"
MJ = r"D:\brick\D\202606\working\something_like MOSIAC\jsonl\methods.jsonl"

rows = json.load(io.open(RES, encoding="utf-8"))["results"]
norm = lambda x: str(x or "").strip().lower().replace(" ", "")


def parts(pred):
    if not isinstance(pred, dict):
        return "", ""
    if "top1_functional" in pred:
        return norm(pred.get("top1_functional")), norm(pred.get("top1_basis"))
    p = pred.get("pred") or {}
    return norm(p.get("functional")), norm(p.get("basis"))


print("=" * 72)
print("(1) SINGLE-FIT RATES -- functional-only and basis-only vs. the scored pair")
print("=" * 72)
for cell in sorted({r["specialist_cell"] for r in rows}):
    sub = [r for r in rows if r["specialist_cell"] == cell]
    print("\n  {0}  (n = {1})".format(cell, len(sub)))
    print("    {0:12s} {1:>8s} {2:>8s} {3:>8s}   {4}".format(
        "condition", "pair", "func", "basis", "func-only gain over pair"))
    for c in ("rag_alone", "sft_alone", "sft_rag"):
        pair = func = bas = 0
        for r in sub:
            tf, tb = norm(r["true_functional"]), norm(r["true_basis"])
            pf, pb = parts(r.get(c))
            func += pf == tf
            bas += pb == tb
            pair += (pf == tf and pb == tb)
        n = len(sub)
        print("    {0:12s} {1:7.1f}% {2:7.1f}% {3:7.1f}%   {4:+.1f} pp".format(
            c, 100 * pair / n, 100 * func / n, 100 * bas / n, 100 * (func - pair) / n))

print()
print("=" * 72)
print("(2) WHAT THE SFT MODEL ACTUALLY EMITS  (fields present in its JSON output)")
print("=" * 72)
cnt = collections.Counter()
tot = 0
for r in rows:
    p = (r.get("sft_alone") or {}).get("pred")
    if isinstance(p, dict):
        tot += 1
        for k in p:
            cnt[k] += 1
print("  parsed SFT predictions: {0}".format(tot))
for k, v in cnt.most_common():
    print("    {0:18s} emitted in {1:5.1f}% of predictions".format(k, 100 * v / tot))
print("\n  SCORED in the paper: functional, basis only.")
print("  The results file carries gold for those two only (true_functional/true_basis),")
print("  so the other five cannot be scored from this artifact.")

print()
print("=" * 72)
print("(3) CORPUS COVERAGE -- how often each parameter is recorded at all")
print("=" * 72)
recs = [json.loads(l) for l in io.open(MJ, encoding="utf-8") if l.strip()]
for f in ["functional", "basis", "scf_convergence", "grid_level", "ri_approx",
          "final_grid_level", "aux_basis", "dispersion", "relativistic",
          "solvent_model", "cbs_scheme"]:
    pop = sum(1 for r in recs if r.get(f) not in (None, "", [], 0))
    bar = "#" * int(40 * pop / len(recs))
    print("  {0:18s} {1:5.1f}%  {2}".format(f, 100 * pop / len(recs), bar))
print("\n  n = {0} sampled records".format(len(recs)))
