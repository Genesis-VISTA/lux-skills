# Sum MFMA counters per kernel-name class from a rocprofv3 counter_collection.csv dir.
import csv, glob, sys, collections
tot = collections.Counter(); per = collections.Counter()
for f in glob.glob(sys.argv[1] + "/**/*counter_collection.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        v = float(r["Counter_Value"]); tot[r["Counter_Name"]] += v
        if v: per[(r["Counter_Name"], r["Kernel_Name"][:70])] += v
for k, v in tot.items(): print(f"TOTAL {k} {v:.4e}")
for (c, k), v in per.most_common(8): print(f"  {c} {v:.4e} {k}")
