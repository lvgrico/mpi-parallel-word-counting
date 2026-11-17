#!/usr/bin/env python3
"""
analyze.py - turns the logs produced by run_all.sh into the tables / metrics
required by the lab and (optionally) injects them into README.md.

Usage (from the repo root, after run_all.sh has finished):
    python analyze.py              # uses the largest np found in results/logs
    python analyze.py --p 4        # choose the process count p of the analysis
    python analyze.py --no-readme  # only write results/RESULTS.md

Metrics:  S_p = T_seq / T_p      E_p = S_p / p
          LI = max(t_i) / mean(t_i)    SPREAD = (max(t_i) - min(t_i)) / mean(t_i)
"""
import argparse
import glob
import os
import re
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "results", "logs")


def mean(x):
    return sum(x) / len(x)


def sd(x):
    return st.stdev(x) if len(x) > 1 else 0.0


def parse_mpi_log(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"EXECUTION_TIME=([\d.]+)", txt)
    if not m:
        return None
    ranks = {}
    for r, f, t in re.findall(r"RANK=(\d+) FILES=(\d+) TOKENS=\d+ LOCAL_TIME=([\d.]+)", txt):
        ranks[int(r)] = (int(f), float(t))
    match = re.search(r"MATCHES_BASELINE=(\w+)", txt)
    times = [t for _, t in ranks.values()]
    return {
        "T": float(m.group(1)),
        "ranks": ranks,
        "LI": max(times) / mean(times),
        "SPREAD": (max(times) - min(times)) / mean(times),
        "match": match.group(1) if match else "unknown",
    }


def load_runs():
    data = defaultdict(list)  # (version, np) -> [run dict]
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "mpi*_np*_run*.log"))):
        m = re.match(r"(mpi\d)_np(\d+)_run(\d+)\.log", os.path.basename(path))
        if not m:
            continue
        r = parse_mpi_log(path)
        if r:
            data[(m.group(1), int(m.group(2)))].append(r)
    return data


def load_baseline():
    vals = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "baseline*.log"))):
        txt = open(path, encoding="utf-8").read()
        m = re.search(r"Tiempo de ejecuci.n:\s*([\d.]+)", txt)
        if m:
            vals.append(float(m.group(1)))
    return vals


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def summarize(runs, tseq, p):
    Ts = [r["T"] for r in runs]
    Tp = mean(Ts)
    Sp = tseq / Tp
    mean_ranks = {}
    for rk in runs[0]["ranks"]:
        mean_ranks[rk] = (runs[0]["ranks"][rk][0], mean([r["ranks"][rk][1] for r in runs]))
    max_local = [max(t for _, t in r["ranks"].values()) for r in runs]
    mean_local = [mean([t for _, t in r["ranks"].values()]) for r in runs]
    return {
        "Ts": Ts, "Tp": Tp, "sd": sd(Ts), "Sp": Sp, "Ep": Sp / p,
        "LI": mean([r["LI"] for r in runs]), "SPREAD": mean([r["SPREAD"] for r in runs]),
        "mean_ranks": mean_ranks,
        "overhead": mean([r["T"] - m for r, m in zip(runs, max_local)]),
        "imb_cost": mean([a - b for a, b in zip(max_local, mean_local)]),
        "mean_local": mean(mean_local),
        "match": all(r["match"] == "True" for r in runs),
    }


def build(data, tseq_list, p, runs_n):
    tseq = mean(tseq_list)
    S = {v: summarize(data[(v, p)], tseq, p) for v in ("mpi1", "mpi2") if data.get((v, p))}
    out = {}

    # ---- section 4 -------------------------------------------------------
    a = f"#### 4a. Sequential baseline timing\n\n" + table(
        ["Run", "T_seq (s)"], [[i + 1, f"{t:.4f}"] for i, t in enumerate(tseq_list)]) + \
        f"\n\n**T_seq = {tseq:.4f} s** (mean of {len(tseq_list)} run(s)).\n"

    def runs_table(v):
        return table(["Run", "Total time (s)", "LI", "Spread"],
                     [[i + 1, f"{r['T']:.4f}", f"{r['LI']:.3f}", f"{r['SPREAD']:.3f}"]
                      for i, r in enumerate(data[(v, p)])])

    b = ""
    if "mpi1" in S:
        s = S["mpi1"]
        b = (f"#### 4b. MPI version 1 timing results (p = {p})\n\n" + runs_table("mpi1") +
             f"\n\n**T_p = {s['Tp']:.4f} s** (std {s['sd']:.4f} s), "
             f"**S_p = {s['Sp']:.3f}**, **E_p = {s['Ep'] * 100:.1f}%**. "
             f"Output identical to the baseline in all runs: **{s['match']}**.\n")

    def ranks_table(s):
        return table(["Rank", "Assigned files", "Mean local time (s)"],
                     [[r, f, f"{t:.4f}"] for r, (f, t) in sorted(s["mean_ranks"].items())])

    c = d = ""
    if "mpi1" in S:
        s = S["mpi1"]
        c = (f"#### 4c. Load imbalance evidence (version 1, p = {p})\n\n" + ranks_table(s) +
             f"\n\nMean local time over the processes: {s['mean_local']:.4f} s. "
             f"**LI = {s['LI']:.3f}**, relative spread = {s['SPREAD']:.3f} "
             f"(mean over {len(data[('mpi1', p)])} runs of the per-run values).\n")
    if "mpi2" in S:
        s = S["mpi2"]
        d = (f"MPI version 2 results (p = {p}):\n\n" + runs_table("mpi2") + "\n\n" +
             ranks_table(s) + f"\n\n**T_p = {s['Tp']:.4f} s** (std {s['sd']:.4f} s), "
             f"**S_p = {s['Sp']:.3f}**, **E_p = {s['Ep'] * 100:.1f}%**, **LI = {s['LI']:.3f}**. "
             f"Output identical to the baseline in all runs: **{s['match']}**.\n")
    if len(S) == 2:
        s1, s2 = S["mpi1"], S["mpi2"]
        cmp_t = table(["Metric", "Sequential", "MPI v1", "MPI v2"], [
            ["Total time T (s)", f"{tseq:.4f}", f"{s1['Tp']:.4f}", f"{s2['Tp']:.4f}"],
            ["Speedup S_p", "1.000", f"{s1['Sp']:.3f}", f"{s2['Sp']:.3f}"],
            ["Efficiency E_p", "-", f"{s1['Ep'] * 100:.1f}%", f"{s2['Ep'] * 100:.1f}%"],
            ["Load imbalance LI", "-", f"{s1['LI']:.3f}", f"{s2['LI']:.3f}"],
            ["Relative spread", "-", f"{s1['SPREAD']:.3f}", f"{s2['SPREAD']:.3f}"],
        ])
        d += f"\n**Comparison (p = {p}):**\n\n{cmp_t}\n"
    # all-np overview
    nps = sorted({k[1] for k in data})
    if len(nps) > 1:
        rows = []
        for n in nps:
            for v in ("mpi1", "mpi2"):
                if data.get((v, n)):
                    q = summarize(data[(v, n)], tseq, n)
                    rows.append([v, n, f"{q['Tp']:.4f}", f"{q['Sp']:.3f}",
                                 f"{q['Ep'] * 100:.1f}%", f"{q['LI']:.3f}"])
        d += "\n**All process counts measured:**\n\n" + table(
            ["Version", "p", "T_p (s)", "S_p", "E_p", "LI"], rows) + "\n"
    out["RESULTS"] = "\n".join(x for x in (a, b, c) if x)
    out["RESULTS_D"] = d

    # ---- section 5 (data-driven drafts) ---------------------------------
    if len(S) == 2:
        s1, s2 = S["mpi1"], S["mpi2"]
        better = "improved on" if s1["Sp"] > 1.05 else "did not meaningfully improve on"
        gap = p - s1["Sp"]
        ideal_t = tseq / p
        a5 = (
            f"**Version 1 {better} the sequential baseline.** With p = {p} it took "
            f"T_p = {s1['Tp']:.3f} s against T_seq = {tseq:.3f} s, a speedup of "
            f"S_p = {s1['Sp']:.2f} versus the ideal {p} (efficiency {s1['Ep'] * 100:.0f}%). "
            f"The ideal time would have been T_seq/p = {ideal_t:.3f} s, so {s1['Tp'] - ideal_t:.3f} s "
            f"are unexplained by perfect scaling. Using the logs, the total time of version 1 splits as:\n\n"
            f"- mean local compute time per process: {s1['mean_local']:.3f} s "
            f"({'higher' if s1['mean_local'] > ideal_t else 'lower'} than T_seq/p = {ideal_t:.3f} s: "
            f"the difference reflects contention for CPU, memory and disk when p processes read files at the same time);\n"
            f"- cost of imbalance, max(t_i) - mean(t_i): {s1['imb_cost']:.3f} s "
            f"(everyone waits for the slowest process);\n"
            f"- non-parallel overhead, T_p - max(t_i): {s1['overhead']:.3f} s "
            f"(MPI start-up/barrier, listing the directory and reading the query in rank 0, "
            f"broadcast, scatter, pickling and gathering of the partial counters, and the merge in rank 0).\n"
        )
        li_drop = (s1["LI"] - s2["LI"])
        t_gain = (s1["Tp"] - s2["Tp"]) / s1["Tp"] * 100
        noise = max(s1["sd"], s2["sd"])
        real = (s1["Tp"] - s2["Tp"]) > max(noise, 0.02 * s1["Tp"])
        verdict = ("**Yes, this is a real improvement**: the reduction of the mean total time is larger than "
                   "the run-to-run standard deviation and than 2% of T_p."
                   if real else
                   "**The improvement is not clearly real**: the difference between the mean total times is "
                   "within the run-to-run noise (standard deviation) or below 2% of T_p, so with this "
                   "experiment it cannot be told apart from measurement variation.")
        b5 = (
            f"**Evidence in version 1.** Every process received the same number of files, but the local times "
            f"in table 4c range from {min(t for _, t in s1['mean_ranks'].values()):.3f} s to "
            f"{max(t for _, t in s1['mean_ranks'].values()):.3f} s, giving LI = {s1['LI']:.3f} "
            f"(spread {s1['SPREAD']:.3f}). The cause is that the corpus has files of very different size "
            f"(about 60% small, 30% medium, 10% large), so equal file counts are not equal work, and the total "
            f"time is set by the slowest process.\n\n"
            f"**Effect of version 2.** Assigning files by size (LPT) changed LI from {s1['LI']:.3f} to "
            f"{s2['LI']:.3f} (reduction of {li_drop:.3f}) and T_p from {s1['Tp']:.3f} s to {s2['Tp']:.3f} s "
            f"({t_gain:+.1f}% time change; S_p {s1['Sp']:.2f} -> {s2['Sp']:.2f}, "
            f"E_p {s1['Ep'] * 100:.0f}% -> {s2['Ep'] * 100:.0f}%). {verdict} "
            f"The maximum possible gain from balancing is bounded by the imbalance cost of version 1 "
            f"({s1['imb_cost']:.3f} s), which is "
            f"{s1['imb_cost'] / s1['Tp'] * 100:.1f}% of its total time.\n"
        )
        best = "MPI version 2" if s2["Tp"] < s1["Tp"] else "MPI version 1"
        sb = s2 if best == "MPI version 2" else s1
        conc = (
            f"For this problem and dataset I would recommend **{best}**"
            f"{' (size-aware static distribution)' if best == 'MPI version 2' else ' (equal file counts)'}: with "
            f"p = {p} it reached S_p = {sb['Sp']:.2f} and E_p = {sb['Ep'] * 100:.0f}% with LI = {sb['LI']:.3f}, "
            f"compared with S_p = {(s1 if best == 'MPI version 2' else s2)['Sp']:.2f}, "
            f"E_p = {(s1 if best == 'MPI version 2' else s2)['Ep'] * 100:.0f}% and "
            f"LI = {(s1 if best == 'MPI version 2' else s2)['LI']:.3f} for the other one. "
            f"{'The gain is small relative to the noise, so the choice is justified mostly by the fact that version 2 is not slower and costs only a few getsize calls.' if not real else 'The difference exceeds the measurement noise.'} "
            f"Neither version reaches the ideal speedup of {p}; the remaining gap comes from the sequential "
            f"part and communication of the algorithm, resource contention on the machine and the residual "
            f"imbalance, which means that further gains would need to attack I/O and overhead rather than "
            f"only the distribution of files."
        )
        out["ANALYSIS_A"], out["ANALYSIS_B"], out["CONCLUSION"] = a5, b5, conc
    return out


def inject(readme, name, content):
    pat = re.compile(rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
    if not pat.search(readme):
        return readme
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", readme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, help="process count used for the analysis")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()

    data = load_runs()
    tseq_list = load_baseline()
    if not data or not tseq_list:
        raise SystemExit("No logs found in results/logs. Run run_all.sh first.")
    nps = sorted({k[1] for k in data})
    p = args.p or nps[-1]
    out = build(data, tseq_list, p, None)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(out["RESULTS"] + "\n" + out["RESULTS_D"] + "\n\n## Draft analysis\n\n" + out.get("ANALYSIS_A", "") +
                "\n\n" + out.get("ANALYSIS_B", "") + "\n\n## Draft conclusion\n\n" + out.get("CONCLUSION", "") + "\n")
    print(out["RESULTS"] + "\n" + out["RESULTS_D"])
    print("\nSaved results/RESULTS.md")

    readme_path = os.path.join(ROOT, "README.md")
    if not args.no_readme and os.path.isfile(readme_path):
        txt = open(readme_path, encoding="utf-8").read()
        for tag, key in (("RESULTS", "RESULTS"), ("RESULTS_D", "RESULTS_D"), ("ANALYSIS_A", "ANALYSIS_A"),
                         ("ANALYSIS_B", "ANALYSIS_B"), ("CONCLUSION", "CONCLUSION")):
            if key in out:
                txt = inject(txt, tag, out[key])
        open(readme_path, "w", encoding="utf-8").write(txt)
        print("README.md updated (sections 4, 5a, 5b and 6).")


if __name__ == "__main__":
    main()
