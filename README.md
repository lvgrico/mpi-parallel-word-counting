## 1. Title

# Parallel Word Counting in a Text Corpus with MPI

**Lab 2 – Universidad del Norte**  
Author: Laura González Rico

## 2. Problem Description

`generator.py` creates a folder `dataset/` with:

- `consulta.txt`: 200 query words, one per line;
- `file_0001.txt … file_3000.txt`: a corpus of about 45 million words. File sizes are deliberately heterogeneous (≈60 % small, ≈30 % medium and ≈10 % large files).

The goal is to count how many times each word of `consulta.txt` appears in the whole corpus (case-insensitive, whitespace tokenisation) and report the **top 10** most frequent query words. The lab compares a sequential baseline against two MPI implementations to study how parallelisation and the **workload distribution strategy** affect performance:

| Version | File | Distribution strategy |
|---|---|---|
| Baseline | `baseline_secuencial.py` | none (one process) |
| MPI v1 | `mpi1.py` | static, same **number** of files per process |
| MPI v2 | `mpi2.py` | static, same amount of **bytes** per process (size-aware, LPT) |

## 3. Environment and Execution Instructions

Everything runs inside the course Docker image `augustosalazar/slim-mpi:2` (Python + mpi4py + Open MPI). No local installation is needed. Commands are shown for Linux/macOS; on the Windows command prompt replace `"$(pwd)"` with `"%cd%"`.

Repository layout:

```
generator.py            (provided, unchanged)
baseline_secuencial.py  (provided, unchanged)
run_all.sh              (provided, unchanged)
mpi1.py                 MPI version 1
mpi2.py                 MPI version 2
analyze.py              helper: computes T_p, S_p, E_p, LI from the logs
README.md
```

### 3a. Experimental plan

1. Generate the dataset with `generator.py` (fixed seed 42, so every group has the same data).
2. Run the sequential baseline and record `T_seq` and its output.
3. Find the number of processes `p` reported by `Detectar_procesos_mpi` and use that single value for all MPI experiments.
4. Run MPI v1 **3 times** with `p` processes. Record the total execution time and the local time of every process.
5. Run MPI v2 **3 times** with the same `p`.
6. For each version compute: average total time `T_p`, speedup `S_p = T_seq / T_p`, efficiency `E_p = S_p / p`, load imbalance `LI = max(t_i) / mean(t_i)` (and the optional relative spread `(max(t_i) − min(t_i)) / mean(t_i)`), where `t_i` is the local processing time of process *i*. `LI` is computed per run and then averaged.
7. Check that the top 10 (and the full counts) of both MPI versions match the baseline.

Controlled conditions: same dataset, same container, same `p`, same query, nothing else heavy running during the measurements, and the first execution is not discarded (all runs are reported).

### 3b. Sequential baseline (what it does and how)

`baseline_secuencial.py` loads `consulta.txt` into a set, walks over every `file_*.txt` in `dataset/`, lowercases and splits each line into tokens and increments a `Counter` for every token that belongs to the query. It prints the execution time, the number of files/tokens/occurrences and the top 10, and saves all counts in `dataset/baseline_results.csv`.

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

### 3c. MPI version 1 (what it does and how)

`mpi1.py` follows the steps of the lab statement:

1. rank 0 reads `consulta.txt` (same preprocessing as the baseline);
2. rank 0 broadcasts the query words to all processes with `comm.bcast`;
3. rank 0 lists and sorts the `file_*.txt` files;
4. the files are split **statically** in `p` contiguous blocks with the same number of files (±1) and sent with `comm.scatter`;
5. each process counts the occurrences of the query words in its own files with exactly the same counting code as the baseline and measures its **local time** (only reading + counting, no communication);
6. the partial `Counter`s are collected in rank 0 with `comm.gather`;
7. rank 0 merges them, prints the top 10 and checks the result against `dataset/baseline_results.csv`.

For every process the program prints `RANK=<r> FILES=<n> TOKENS=<t> LOCAL_TIME=<s>`, then `LI` and the relative spread, and finally `EXECUTION_TIME=<s>` (wall-clock of the whole parallel program measured in rank 0 after a barrier; includes broadcast, scatter, counting, gather and merge). The `EXECUTION_TIME=` line is the one parsed by `run_all.sh`.

### 3d. Test procedure

*Correctness:* `mpi1.py` and `mpi2.py` print `MATCHES_BASELINE=True` when the merged global counts are identical to `baseline_results.csv` (full dictionary comparison, which is stricter than comparing only the top 10; ties in the top 10 may be listed in a different order).

*Experiments:* the provided `run_all.sh` regenerates the dataset, runs the baseline, and then runs both MPI programs `RUNS` times for every value in `NP_LIST`. Use the `p` reported by `Detectar_procesos_mpi` (example with `p = 4`):

```bash
docker run --rm -v "$(pwd)":/app -e NP_LIST="4" -e RUNS=3 \
    augustosalazar/slim-mpi:2 sh /app/run_all.sh

# compute T_p, S_p, E_p, LI and fill this README (sections 4, 5a, 5b and 6)
python analyze.py --p 4
```

Logs are stored in `results/logs/` and the raw times in `results/times_summary.csv`. A single run of one program can also be launched with:

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 \
    mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi1.py
```

## 4. Experimental Plan Execution

All numbers in this section were produced by `run_all.sh` and `analyze.py` (see `results/`).

<!-- AUTO:RESULTS -->
#### 4a. Sequential baseline timing

| Run | T_seq (s) |
|---|---|
| 1 | 26.0952 |

**T_seq = 26.0952 s** (mean of 1 run(s)).

#### 4b. MPI version 1 timing results (p = 16)

| Run | Total time (s) | LI | Spread |
|---|---|---|---|
| 1 | 5.0523 | 1.123 | 0.261 |
| 2 | 4.8509 | 1.124 | 0.246 |
| 3 | 5.0148 | 1.123 | 0.243 |

**T_p = 4.9726 s** (std 0.1071 s), **S_p = 5.248**, **E_p = 32.8%**. Output identical to the baseline in all runs: **True**.

#### 4c. Load imbalance evidence (version 1, p = 16)

| Rank | Assigned files | Mean local time (s) |
|---|---|---|
| 0 | 188 | 4.4943 |
| 1 | 188 | 4.6855 |
| 2 | 188 | 4.2569 |
| 3 | 188 | 3.8552 |
| 4 | 188 | 4.9426 |
| 5 | 188 | 4.2449 |
| 6 | 188 | 4.5251 |
| 7 | 188 | 4.6317 |
| 8 | 187 | 4.1163 |
| 9 | 187 | 4.5229 |
| 10 | 187 | 4.2786 |
| 11 | 187 | 4.4293 |
| 12 | 187 | 4.6771 |
| 13 | 187 | 3.8567 |
| 14 | 187 | 4.1699 |
| 15 | 187 | 4.7026 |

Mean local time over the processes: 4.3994 s. **LI = 1.123**, relative spread = 0.250 (mean over 3 runs of the per-run values).

<!-- /AUTO:RESULTS -->

#### 4d. MPI version 2: how it corrects the imbalance, and its results

Version 1 balances the **number** of files, but processing time is roughly proportional to the **size** of each file, and the sizes of this corpus are very uneven. A process that happens to receive many large files finishes late while the others wait.

`mpi2.py` keeps everything from version 1 (broadcast of the query, local counting, gather and merge) and only changes how rank 0 builds the blocks before the `scatter`:

1. the size in bytes of every file is read with `os.path.getsize` (metadata only, the files are not read). To keep this cheap, each process reads the sizes of a round-robin slice of the files and rank 0 gathers them;
2. rank 0 sorts the files by size, largest first;
3. each file is assigned to the process whose accumulated size is currently the smallest (greedy *Longest-Processing-Time-first*, implemented with a heap).

LPT is a classic heuristic that keeps the difference between process loads on the order of the size of a single file, so the per-process work becomes almost equal even though the number of files per process may now differ. The assignment is still static, so no communication is added during the computation, and the assignment itself costs only O(n log n) on 3000 files in rank 0. The time spent reading sizes and building the assignment is included in `T_p`, so it is paid for by version 2. Results of version 2:

<!-- AUTO:RESULTS_D -->
MPI version 2 results (p = 16):

| Run | Total time (s) | LI | Spread |
|---|---|---|---|
| 1 | 5.0299 | 1.019 | 0.039 |
| 2 | 5.5855 | 1.021 | 0.045 |
| 3 | 5.5094 | 1.014 | 0.031 |

| Rank | Assigned files | Mean local time (s) |
|---|---|---|
| 0 | 189 | 4.7258 |
| 1 | 188 | 4.6881 |
| 2 | 188 | 4.6501 |
| 3 | 186 | 4.6229 |
| 4 | 187 | 4.6970 |
| 5 | 188 | 4.6288 |
| 6 | 188 | 4.6407 |
| 7 | 187 | 4.7018 |
| 8 | 187 | 4.6464 |
| 9 | 188 | 4.7366 |
| 10 | 187 | 4.6991 |
| 11 | 187 | 4.6787 |
| 12 | 188 | 4.7077 |
| 13 | 187 | 4.6547 |
| 14 | 188 | 4.6636 |
| 15 | 187 | 4.6855 |

**T_p = 5.3749 s** (std 0.3012 s), **S_p = 4.855**, **E_p = 30.3%**, **LI = 1.018**. Output identical to the baseline in all runs: **True**.

**Comparison (p = 16):**

| Metric | Sequential | MPI v1 | MPI v2 |
|---|---|---|---|
| Total time T (s) | 26.0952 | 4.9726 | 5.3749 |
| Speedup S_p | 1.000 | 5.248 | 4.855 |
| Efficiency E_p | - | 32.8% | 30.3% |
| Load imbalance LI | - | 1.123 | 1.018 |
| Relative spread | - | 0.250 | 0.038 |

<!-- /AUTO:RESULTS_D -->

## 5. Analysis

### 5a. Parallel performance

<!-- AUTO:ANALYSIS_A -->
**Version 1 improved on the sequential baseline.** With p = 16 it took T_p = 4.973 s against T_seq = 26.095 s, a speedup of S_p = 5.25 versus the ideal 16 (efficiency 33%). The ideal time would have been T_seq/p = 1.631 s, so 3.342 s are unexplained by perfect scaling. Using the logs, the total time of version 1 splits as:

- mean local compute time per process: 4.399 s (higher than T_seq/p = 1.631 s: the difference reflects contention for CPU, memory and disk when p processes read files at the same time);
- cost of imbalance, max(t_i) - mean(t_i): 0.543 s (everyone waits for the slowest process);
- non-parallel overhead, T_p - max(t_i): 0.030 s (MPI barrier, listing the directory and reading the query in rank 0, broadcast, scatter, pickling and gathering of the partial counters, the merge in rank 0, and file-system metadata calls, which can be slow on folders shared with Docker on Windows/macOS).

Decomposing the 3.342 s gap between T_p and the ideal T_seq/p = 1.631 s: slower per-process compute 2.768 s (83%); load imbalance 0.543 s (16%); non-parallel overhead 0.030 s (1%). In other words, most of the gap (83%) is that each process needs much more time than T_seq/p to count its share (contention for CPU cores, memory bandwidth and file-system I/O, and possibly more logical processors than physical cores), not communication or serial work, which are small.

<!-- /AUTO:ANALYSIS_A -->

Four things can separate a measured speedup from the ideal one: (i) the sequential part of the program (reading the query, listing the directory, building blocks, merging counters in rank 0), which does not shrink with `p` (Amdahl's law); (ii) the processes share the same disk, page cache and memory bandwidth, and the work is I/O-heavy; (iii) if `p` is larger than the number of physical cores, processes compete for CPU; and (iv) the slowest process determines the end of the program. The decomposition above shows which ones dominate here: a large "slower per-process compute" share points to (ii) and (iii), a large overhead share points to (i), and the imbalance share is (iv).

### 5b. Load balance

<!-- AUTO:ANALYSIS_B -->
**Evidence in version 1.** Every process received the same number of files, but the local times in table 4c range from 3.855 s to 4.943 s, giving LI = 1.123 (spread 0.250). The cause is that the corpus has files of very different size (about 60% small, 30% medium, 10% large), so equal file counts are not equal work, and the total time is set by the slowest process.

**Effect of version 2.** Assigning files by size (LPT) changed LI from 1.123 to 1.018 (reduction of 0.105) and T_p from 4.973 s to 5.375 s (+8.1% change in total time; S_p 5.25 -> 4.86, E_p 33% -> 30%). **Version 2 was slower than version 1, even though its LI is lower.** The difference is larger than the run-to-run noise, so balancing the local times did not translate into a better total time. Comparing T_p - max(t_i) of both versions (overhead 0.030 s in v1 vs 0.614 s in v2) shows where the time went: the extra work that version 2 does outside the counting loop (reading file sizes, gathering them and building the assignment) cost more than the 0.459 s of imbalance it removed. The maximum possible gain from balancing is bounded by the imbalance cost of version 1 (0.543 s), which is 10.9% of its total time.

<!-- /AUTO:ANALYSIS_B -->

*Caveat:* with only three runs per version, the gap between versions (about 0.4 s) is of the same order as the run-to-run variation of version 2 (its first run took 5.03 s, as fast as version 1). The result that version 2 is not faster is therefore suggestive rather than conclusive; what is solid is that balancing the local times (LI 1.12 → 1.02) did not produce a visible gain in total time.

### 5c. Limitations

- **Single machine / container.** All "processes" share the same CPU cores, memory and disk, and run inside Docker (possibly inside a VM on Windows/macOS). Results show behaviour on one node and say nothing about distributed-memory clusters, where communication cost is much higher.
- **Only one value of `p`.** The lab fixes `p` to the number reported by `Detectar_procesos_mpi`, so scalability (how S_p and E_p change with p) cannot be concluded.
- **Few repetitions.** Three runs per configuration give a rough mean; the standard deviation is reported but is itself estimated from three samples. Small differences (a few percent) between versions cannot be considered significant.
- **File-system overhead under Docker.** The dataset is a bind-mounted folder. On Docker Desktop for Windows/macOS, metadata calls (listing, `stat`, `getsize`) and file opens on bind mounts are much slower than on a native Linux disk, so I/O and metadata costs weigh more than they would on a cluster or a Linux host. In a first implementation, listing the files with one `os.path.isfile` call per file (3000 calls) and reading all sizes sequentially in rank 0 added about 4.6 s (v1) and 10.8 s (v2) of overhead on this setup, which made version 2 look 65% slower (9.7 s vs 16.0 s). Both were fixed (`os.scandir` for listing, sizes read in parallel by all ranks), which brought the overhead down to 0.03 s (v1) and 0.6 s (v2). This shows how strongly this environment penalises metadata calls, and the remaining differences between versions should be read with that in mind.
- **The provided baseline uses per-file `stat` calls that the MPI versions avoid.** `baseline_secuencial.py` (unchanged, as required) calls `os.path.isfile` for each of the 3000 files, while the MPI versions list files with `os.scandir`. Since the first-run measurements suggest these calls cost several seconds on this Docker setup, `T_seq` includes a cost that the MPI versions do not pay, and the reported speedups are somewhat optimistic.
- **Baseline measured once.** `run_all.sh` runs the sequential baseline a single time, so `T_seq` has no estimate of its own variability, and every speedup inherits that uncertainty.
- **Disk cache.** After the first run the corpus is probably in the operating-system page cache, so I/O is cheaper than on a cold start and the first run of each experiment may be slower.
- **Noise from other programs** on the host (browser, antivirus, background services) and CPU frequency scaling were not controlled.
- **One dataset.** The data are synthetic and generated with a fixed seed; the size distribution (60/30/10 %) was designed to create imbalance. Real corpora may be more or less skewed.
- **File size is only a proxy of the work.** LPT uses the number of bytes as an estimate of the processing cost; it ignores the content of the files and cache effects.
- **Timer scope.** `T_p` includes MPI synchronisation inside the program but not the start-up of `mpirun`/Python or the Docker container, while `T_seq` of the baseline includes only the counting itself, so the speedup is slightly pessimistic or optimistic depending on that overhead.

Overall, these limitations do not change the qualitative conclusions (whether the MPI versions are faster than the baseline and whether LI decreases), but they weaken any claim based on small differences in `T_p` and any extrapolation to other numbers of processes, other machines or other datasets.

## 6. Conclusions

<!-- AUTO:CONCLUSION -->
For this problem and dataset I would recommend **MPI version 1** (equal file counts): with p = 16 it reached S_p = 5.25 and E_p = 33% with LI = 1.123, compared with S_p = 4.86, E_p = 30% and LI = 1.018 for the other one. A better LI alone did not pay off: the lower imbalance of the other version was not worth its extra overhead in this environment. Neither version reaches the ideal speedup of 16: for version 1, most of the gap (83%) is that each process needs much more time than T_seq/p to count its share (contention for CPU cores, memory bandwidth and file-system I/O, and possibly more logical processors than physical cores), not communication or serial work, which are small. Further gains would therefore have to come from reducing contention and I/O cost per process rather than only from the distribution of files.
<!-- /AUTO:CONCLUSION -->
