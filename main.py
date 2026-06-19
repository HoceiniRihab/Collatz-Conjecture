import time
import numpy as np

from sequential_waves import build_odd_matrix, extract_form_indices, report_max_value
from parallel_waves import (
    _build_A_array,
    build_odd_matrix_gpu,
    report_max_value_gpu,
    extract_form_indices_gpu,
)
from plots import plot_scaling_comparison, DATA_PARALLEL, DATA_SEQUENTIAL


n = 500   # rows
m = 500   # columns


if __name__ == "__main__":

    print(f"Matrix: {n} rows × {m} cols\n")

    # ── Sequential waves ────────────────────────────────────
    t0 = time.perf_counter()
    matrix = build_odd_matrix(n_rows=n, n_cols=m)
    max_val, k, alpha, mult = report_max_value(matrix)
    positions, n_indices = extract_form_indices(matrix)
    t1 = time.perf_counter()

    print("── Sequential (CPU) ──────────────────────────")
    print(f"  N ≈ 2^{k} × {mult:.2f}   (log₂ = {alpha:.2f})")
    print(f"  Elements satisfying 3(2n+1): {len(positions)}")
    print(f"  Execution time: {t1 - t0:.4f} s\n")

    # ── Parallel waves ──────────────────────────────────────
    A = _build_A_array(m)
    t0 = time.perf_counter()
    mod6_gpu, log_gpu, mod6_host, log_host = build_odd_matrix_gpu(n_rows=n, n_cols=m)
    max_val, k, alpha, mult = report_max_value_gpu(log_gpu, n, m, A)
    positions, log_n_values = extract_form_indices_gpu(mod6_gpu, log_gpu, n, m)
    t1 = time.perf_counter()

    print("── Parallel (GPU) ────────────────────────────")
    print(f"  Elements satisfying 3(2n+1): {len(positions)}")
    print(f"  Execution time: {t1 - t0:.4f} s\n")

    # ── Scalability plots ────────────────────────────────────
    plot_scaling_comparison(
        data_par  = DATA_PARALLEL,
        data_seq  = DATA_SEQUENTIAL,
        save_path = "rihab_scaling_comparison.png",
    )
