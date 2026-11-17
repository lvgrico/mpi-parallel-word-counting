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
| 1 | 27.4350 |

**T_seq = 27.4350 s** (mean of 1 run(s)).

#### 4b. MPI version 1 timing results (p = 16)

| Run | Total time (s) | LI | Spread |
|---|---|---|---|
| 1 | 9.4280 | 1.130 | 0.250 |
| 2 | 9.5204 | 1.132 | 0.244 |
| 3 | 10.1446 | 1.133 | 0.269 |

**T_p = 9.6977 s** (std 0.3898 s), **S_p = 2.829**, **E_p = 17.7%**. Output identical to the baseline in all runs: **True**.

#### 4c. Load imbalance evidence (version 1, p = 16)

| Rank | Assigned files | Mean local time (s) |
|---|---|---|
| 0 | 188 | 4.5927 |
| 1 | 188 | 4.7791 |
| 2 | 188 | 4.3435 |
| 3 | 188 | 3.9610 |
| 4 | 188 | 5.0910 |
| 5 | 188 | 4.3086 |
| 6 | 188 | 4.6311 |
| 7 | 188 | 4.7434 |
| 8 | 187 | 4.2068 |
| 9 | 187 | 4.6023 |
| 10 | 187 | 4.3647 |
| 11 | 187 | 4.5192 |
| 12 | 187 | 4.7890 |
| 13 | 187 | 3.9537 |
| 14 | 187 | 4.2697 |
| 15 | 187 | 4.8227 |

Mean local time over the processes: 4.4987 s. **LI = 1.132**, relative spread = 0.254 (mean over 3 runs of the per-run values).

<!-- /AUTO:RESULTS -->

#### 4d. MPI version 2: how it corrects the imbalance, and its results

Version 1 balances the **number** of files, but processing time is roughly proportional to the **size** of each file, and the sizes of this corpus are very uneven. A process that happens to receive many large files finishes late while the others wait.

`mpi2.py` keeps everything from version 1 (broadcast of the query, local counting, gather and merge) and only changes how rank 0 builds the blocks before the `scatter`:

1. rank 0 reads the size in bytes of every file with `os.path.getsize` (metadata only, the files are not read);
2. files are sorted by size, largest first;
3. each file is assigned to the process whose accumulated size is currently the smallest (greedy *Longest-Processing-Time-first*, implemented with a heap).

LPT is a classic heuristic that keeps the difference between process loads on the order of the size of a single file, so the per-process work becomes almost equal even though the number of files per process may now differ. The assignment is still static, so no communication is added during the computation, and the cost is only O(n log n) on 3000 files in rank 0. Results of version 2:

<!-- AUTO:RESULTS_D -->
MPI version 2 results (p = 16):

| Run | Total time (s) | LI | Spread |
|---|---|---|---|
| 1 | 15.5004 | 1.024 | 0.047 |
| 2 | 15.6181 | 1.030 | 0.041 |
| 3 | 16.8443 | 1.014 | 0.034 |

| Rank | Assigned files | Mean local time (s) |
|---|---|---|
| 0 | 189 | 5.0789 |
| 1 | 188 | 5.0207 |
| 2 | 188 | 5.0346 |
| 3 | 186 | 5.0378 |
| 4 | 187 | 5.0098 |
| 5 | 188 | 5.0245 |
| 6 | 188 | 4.9598 |
| 7 | 187 | 5.0188 |
| 8 | 187 | 5.0108 |
| 9 | 188 | 4.9965 |
| 10 | 187 | 4.9698 |
| 11 | 187 | 4.9976 |
| 12 | 188 | 4.9926 |
| 13 | 187 | 5.0148 |
| 14 | 188 | 5.0976 |
| 15 | 187 | 5.0158 |

**T_p = 15.9876 s** (std 0.7443 s), **S_p = 1.716**, **E_p = 10.7%**, **LI = 1.023**. Output identical to the baseline in all runs: **True**.

**Comparison (p = 16):**

| Metric | Sequential | MPI v1 | MPI v2 |
|---|---|---|---|
| Total time T (s) | 27.4350 | 9.6977 | 15.9876 |
| Speedup S_p | 1.000 | 2.829 | 1.716 |
| Efficiency E_p | - | 17.7% | 10.7% |
| Load imbalance LI | - | 1.132 | 1.023 |
| Relative spread | - | 0.254 | 0.041 |

<!-- /AUTO:RESULTS_D -->

## 5. Analysis

### 5a. Parallel performance

<!-- AUTO:ANALYSIS_A -->
**Version 1 improved on the sequential baseline.** With p = 16 it took T_p = 9.698 s against T_seq = 27.435 s, a speedup of S_p = 2.83 versus the ideal 16 (efficiency 18%). The ideal time would have been T_seq/p = 1.715 s, so 7.983 s are unexplained by perfect scaling. Using the logs, the total time of version 1 splits as:

- mean local compute time per process: 4.499 s (higher than T_seq/p = 1.715 s: the difference reflects contention for CPU, memory and disk when p processes read files at the same time);
- cost of imbalance, max(t_i) - mean(t_i): 0.592 s (everyone waits for the slowest process);
- non-parallel overhead, T_p - max(t_i): 4.607 s (MPI start-up/barrier, listing the directory and reading the query in rank 0, broadcast, scatter, pickling and gathering of the partial counters, and the merge in rank 0).

<!-- /AUTO:ANALYSIS_A -->

The gap to the ideal speedup is explained by the factors measured above and by structural reasons: (i) the sequential part of the program (listing the directory, reading the query, building blocks, merging counters in rank 0) does not shrink with `p` (Amdahl's law); (ii) the work is I/O-heavy and the processes share the same disk, page cache and memory bandwidth; (iii) if `p` is larger than the number of physical cores, processes compete for CPU; (iv) the slowest process determines the end of the program.

### 5b. Load balance

<!-- AUTO:ANALYSIS_B -->
**Evidence in version 1.** Every process received the same number of files, but the local times in table 4c range from 3.954 s to 5.091 s, giving LI = 1.132 (spread 0.254). The cause is that the corpus has files of very different size (about 60% small, 30% medium, 10% large), so equal file counts are not equal work, and the total time is set by the slowest process.

**Effect of version 2.** Assigning files by size (LPT) changed LI from 1.132 to 1.023 (reduction of 0.109) and T_p from 9.698 s to 15.988 s (-64.9% time change; S_p 2.83 -> 1.72, E_p 18% -> 11%). **The improvement is not clearly real**: the difference between the mean total times is within the run-to-run noise (standard deviation) or below 2% of T_p, so with this experiment it cannot be told apart from measurement variation. The maximum possible gain from balancing is bounded by the imbalance cost of version 1 (0.592 s), which is 6.1% of its total time.

<!-- /AUTO:ANALYSIS_B -->

### 5c. Limitations

- **Single machine / container.** All "processes" share the same CPU cores, memory and disk, and run inside Docker (possibly inside a VM on Windows/macOS). Results show behaviour on one node and say nothing about distributed-memory clusters, where communication cost is much higher.
- **Only one value of `p`.** The lab fixes `p` to the number reported by `Detectar_procesos_mpi`, so scalability (how S_p and E_p change with p) cannot be concluded.
- **Few repetitions.** Three runs per configuration give a rough mean; the standard deviation is reported but is itself estimated from three samples. Small differences (a few percent) between versions cannot be considered significant.
- **Disk cache.** After the first run the corpus is probably in the operating-system page cache, so I/O is cheaper than on a cold start and the first run of each experiment may be slower.
- **Noise from other programs** on the host (browser, antivirus, background services) and CPU frequency scaling were not controlled.
- **One dataset.** The data are synthetic and generated with a fixed seed; the size distribution (60/30/10 %) was designed to create imbalance. Real corpora may be more or less skewed.
- **File size is only a proxy of the work.** LPT uses the number of bytes as an estimate of the processing cost; it ignores the content of the files and cache effects.
- **Timer scope.** `T_p` includes MPI synchronisation inside the program but not the start-up of `mpirun`/Python or the Docker container, while `T_seq` of the baseline includes only the counting itself, so the speedup is slightly pessimistic or optimistic depending on that overhead.

Overall, these limitations do not change the qualitative conclusions (whether the MPI versions are faster than the baseline and whether LI decreases), but they weaken any claim based on small differences in `T_p` and any extrapolation to other numbers of processes, other machines or other datasets.

## 6. Conclusions

<!-- AUTO:CONCLUSION -->
For this problem and dataset I would recommend **MPI version 1** (equal file counts): with p = 16 it reached S_p = 2.83 and E_p = 18% with LI = 1.132, compared with S_p = 1.72, E_p = 11% and LI = 1.023 for the other one. The gain is small relative to the noise, so the choice is justified mostly by the fact that version 2 is not slower and costs only a few getsize calls. Neither version reaches the ideal speedup of 16; the remaining gap comes from the sequential part and communication of the algorithm, resource contention on the machine and the residual imbalance, which means that further gains would need to attack I/O and overhead rather than only the distribution of files.
<!-- /AUTO:CONCLUSION -->
