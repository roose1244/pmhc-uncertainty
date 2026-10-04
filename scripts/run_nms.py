import re, sys, time, pathlib, subprocess, tempfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

TOOL = str(pathlib.Path.home() / "tools/netMHCstabpan-1.0/netMHCstabpan")
MODE = sys.argv[1] if len(sys.argv) > 1 else "test"      # "test" or "full"

m = pd.read_parquet("data/processed/master.parquet")

if MODE == "test":
    vc = m.hla.value_counts()
    pick = list(vc.index[:5]) + list(vc.index[-3:])      # 5 common + 3 rarest alleles
    df = m[m.hla.isin(pick)].sample(frac=1, random_state=0).groupby("hla").head(40).copy()
else:
    df = m
tag = "_test" if MODE == "test" else ""

ROW  = re.compile(r"^\s*\d+\s+(HLA-\S+)\s+([A-Z]+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
DIST = re.compile(r"Distance to traning data\s+([\d.]+)\s+\(using nearest neighbor (\S+)\)")
dtu  = lambda h: h.replace("*", "")                      # HLA-A*02:01 -> HLA-A02:01

def run(job):
    hla, length, peps = job
    with tempfile.TemporaryDirectory() as td:
        pf = pathlib.Path(td) / "in.pep"
        pf.write_text("\n".join(peps) + "\n")
        try:
            r = subprocess.run([TOOL, "-p", "-a", dtu(hla), "-l", str(length), "-f", str(pf)],
                               capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return hla, length, [], None, "timeout"
    rows, dist = [], None
    for line in r.stdout.splitlines():
        mt = ROW.match(line)
        if mt:
            rows.append(dict(hla=hla, peptide=mt.group(2),
                             nms_half_life=float(mt.group(4)), nms_rank=float(mt.group(5))))
        d = DIST.search(line)
        if d:
            dist = dict(hla=hla, distance=float(d.group(1)), nearest=d.group(2))
    err = None if rows else ((r.stderr or r.stdout)[-200:] or "no rows parsed")
    return hla, length, rows, dist, err

jobs = [(hla, int(L), g.peptide.unique().tolist())
        for (hla, L), g in df.groupby(["hla", df.peptide.str.len()])]
print(f"{MODE}: {len(df)} rows, {len(jobs)} jobs", flush=True)

t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(run, jobs))
print(f"done in {time.time()-t0:.0f}s")

preds  = [r for _, _, rows, _, _ in results for r in rows]
dists  = [d for _, _, _, d, _ in results if d]
failed = [(h, L, e) for h, L, _, _, e in results if e]

p = pd.DataFrame(preds).drop_duplicates(["hla", "peptide"])
out = df[["id", "peptide", "hla"]].merge(p, on=["peptide", "hla"], how="left")
out["nms_status"] = out.nms_half_life.notna().map({True: "ok", False: "failed"})
out.loc[(out.nms_status == "ok") & (out.peptide.str.len() != 9), "nms_status"] = "approximate_length"
out["nms_source"] = "dtu_standalone"

out[["id", "nms_half_life", "nms_rank"]].to_parquet(f"data/external/netmhcstabpan_preds{tag}.parquet", index=False)
out[["id", "nms_status", "nms_source"]].to_parquet(f"data/external/netmhcstabpan_status{tag}.parquet", index=False)
pd.DataFrame(dists).drop_duplicates("hla").to_parquet(f"data/external/nms_hla_distance{tag}.parquet", index=False)
pd.DataFrame(failed, columns=["hla", "length", "error"]).to_csv(f"data/external/nms_failed{tag}.csv", index=False)

print(out.nms_status.value_counts().to_string())
print("failed jobs:", len(failed))
print(out.sort_values("id").head(8).to_string(index=False))
