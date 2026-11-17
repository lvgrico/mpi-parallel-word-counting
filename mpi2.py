"""
MPI version 2: broadcast of the query + SIZE-AWARE (LPT) file distribution.

Version 1 gives every process the same NUMBER of files, but the corpus has very
heterogeneous file sizes (about 60% small, 30% medium, 10% large), so the work
per process is uneven. Here rank 0 reads the size of every file (os.path.getsize,
a cheap metadata call) and assigns files with the Longest-Processing-Time-first
greedy heuristic: files are sorted by size (descending) and each one goes to the
process with the smallest accumulated size so far. The assignment is still
computed once, before processing starts, so there is no extra communication
during the computation. Counting and gathering are identical to version 1.

Run:  mpirun -np <p> python mpi2.py
"""
import heapq
import os
import sys
from collections import Counter

from mpi4py import MPI # type: ignore

CONSULTA_NAME = "consulta.txt"
CASE_SENSITIVE = False
TOP_N = 10


def cargar_consulta(path, case_sensitive=False):
    """Same behaviour as the sequential baseline."""
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not case_sensitive:
        words = [w.lower() for w in words]
    return set(words)


def listar_archivos(dataset_dir):
    files = [
        f for f in os.listdir(dataset_dir)
        if f.startswith("file_") and f.endswith(".txt")
        and os.path.isfile(os.path.join(dataset_dir, f))
    ]
    return sorted(files)


def partir_por_tamano(dataset_dir, files, p):
    """LPT greedy: biggest file first, always to the least-loaded process.
    The cost of a file is approximated by its size in bytes."""
    sizes = [(os.path.getsize(os.path.join(dataset_dir, f)), f) for f in files]
    sizes.sort(key=lambda x: (-x[0], x[1]))
    heap = [(0, i) for i in range(p)]          # (accumulated bytes, process)
    heapq.heapify(heap)
    blocks = [[] for _ in range(p)]
    for sz, fname in sizes:
        load, i = heapq.heappop(heap)
        blocks[i].append(fname)
        heapq.heappush(heap, (load + sz, i))
    return blocks


def contar_local(dataset_dir, files, targets, case_sensitive=False):
    """Same counting logic as the baseline, restricted to `files`."""
    freq = Counter()
    tokens = 0
    for fname in files:
        with open(os.path.join(dataset_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                if not case_sensitive:
                    words = [w.lower() for w in words]
                tokens += len(words)
                for w in words:
                    if w in targets:
                        freq[w] += 1
    return freq, tokens


def cargar_baseline_csv(path):
    if not os.path.isfile(path):
        return None
    res = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            w, c = line.strip().rsplit(",", 1)
            res[w] = int(c)
    return res


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    comm.Barrier()                      # common starting line for every process
    t_start = MPI.Wtime()

    # 1) rank 0 reads the query, 3) lists the files and 4) builds the blocks
    words, blocks = None, None
    if rank == 0:
        if not os.path.isdir(dataset_dir):
            print(f"Error: dataset folder not found: {dataset_dir}", flush=True)
            comm.Abort(1)
        words = sorted(cargar_consulta(os.path.join(dataset_dir, CONSULTA_NAME),
                                       CASE_SENSITIVE))
        blocks = partir_por_tamano(dataset_dir, listar_archivos(dataset_dir), size)

    # 2) broadcast the query words; 4) static distribution of the files
    words = comm.bcast(words, root=0)
    my_files = comm.scatter(blocks, root=0)
    targets = set(words)

    # 5) local counting (this is the "local processing time")
    t0 = MPI.Wtime()
    local_freq, local_tokens = contar_local(dataset_dir, my_files, targets, CASE_SENSITIVE)
    local_time = MPI.Wtime() - t0

    # 6) partial results gathered in rank 0
    gathered = comm.gather((rank, len(my_files), local_time, local_tokens, local_freq), root=0)

    if rank == 0:
        # 7) global result
        global_freq = Counter()
        for _, _, _, _, freq in gathered:
            global_freq.update(freq)
        total_time = MPI.Wtime() - t_start

        times = [g[2] for g in gathered]
        mean_t = sum(times) / len(times)
        li = max(times) / mean_t if mean_t > 0 else 1.0
        spread = (max(times) - min(times)) / mean_t if mean_t > 0 else 0.0

        print(f"MPI version 2 (static, size-aware LPT) | processes: {size}")
        print("Per-process report:")
        for r, n_files, t, n_tok, _ in sorted(gathered):
            print(f"RANK={r} FILES={n_files} TOKENS={n_tok} LOCAL_TIME={t:.6f}")
        print(f"LI={li:.4f}")
        print(f"SPREAD={spread:.4f}")

        print(f"\nFiles processed: {sum(g[1] for g in gathered)}")
        print(f"Total tokens read: {sum(g[3] for g in gathered)}")
        print(f"Total occurrences found: {sum(global_freq.values())}")
        print(f"\nTop {TOP_N} query words in the corpus:")
        for w, c in global_freq.most_common(TOP_N):
            print(f"  {w}: {c}")

        base = cargar_baseline_csv(os.path.join(dataset_dir, "baseline_results.csv"))
        if base is None:
            print("\nMATCHES_BASELINE=unknown (run baseline_secuencial.py first)")
        else:
            print(f"\nMATCHES_BASELINE={dict(global_freq) == base}")

        print(f"EXECUTION_TIME={total_time:.6f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
