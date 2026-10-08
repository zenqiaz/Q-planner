"""Per-cell field concentration, read from the SFT assistant targets themselves."""
import io, json, os, collections

L = r"D:\brick\D\20260217\working\something_like_MOSIAC\lora_train\jsonl"
FIELDS = ["functional", "basis", "aux_basis", "ri_approx", "grid_level",
          "final_grid_level", "scf_convergence", "dispersion"]


def targets(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        msg = d.get("messages") or []
        asst = next((m.get("content") for m in msg if m.get("role") == "assistant"), None)
        if not asst:
            continue
        try:
            out.append(json.loads(asst))
        except Exception:
            pass
    return out


def stats(recs, f):
    c = collections.Counter()
    for r in recs:
        v = r.get(f)
        if v not in (None, "", [], 0):
            c[str(v)] += 1
    n = sum(c.values())
    if not n:
        return 0, 0.0, 0.0, "-"
    eff = 1.0 / sum((v / n) ** 2 for v in c.values())
    top, tn = c.most_common(1)[0]
    return n, 100.0 * tn / n, eff, top


groups = {
    "organic_general (3,164)": ["sft_organic_general", "sft_val_organic_general"],
    "metal_general (3,870)":   ["sft_metal_general", "sft_val_metal_general"],
    "organic_TDDFT (4,008)":   ["sft_organic_TDDFT", "sft_val_organic_TDDFT"],
}
loaded = {}
for g, files in groups.items():
    recs = []
    for f in files:
        recs += targets(os.path.join(L, f + ".jsonl"))
    loaded[g] = recs
    print("  {0:26s} parsed targets: {1}".format(g, len(recs)))
both = loaded["organic_general (3,164)"] + loaded["metal_general (3,870)"]
allc = both + loaded["organic_TDDFT (4,008)"]
print("  {0:26s} parsed targets: {1}".format("two evaluated cells", len(both)))
print()

for f in FIELDS:
    print("  " + f)
    for label, recs in (("organic_general", loaded["organic_general (3,164)"]),
                        ("metal_general", loaded["metal_general (3,870)"]),
                        ("organic_TDDFT", loaded["organic_TDDFT (4,008)"]),
                        ("-> two evaluated", both),
                        ("-> all three", allc)):
        n, modal, eff, top = stats(recs, f)
        tot = len(recs)
        print("    {0:18s} recorded {1:5d}/{2:5d} ({3:5.1f}%)  modal {4:5.1f}%  eff {5:5.2f}  {6}".format(
            label, n, tot, 100.0 * n / max(1, tot), modal, eff, str(top)[:22]))
    print()
