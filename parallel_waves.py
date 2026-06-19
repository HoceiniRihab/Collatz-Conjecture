import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


cuda_code = r"""
#include <math.h>
#include <float.h>

__global__ void build_matrix_kernel(
    signed char *mod6_mat,   /* [n_rows * n_cols]  int8   */
    double      *log_mat,    /* [n_rows * n_cols]  float64 */
    int n_rows, int n_cols)
{
    /* 2-D grid: x → column j,  y → row i */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n_cols || i >= n_rows) return;

    int count = -1;
    long long Aj = 3;
    while (count < j) {
        Aj += 2;
        if (Aj % 3 != 0) count++;
    }
    int    Aj_mod6 = (int)(Aj % 6);
    double log_Aj  = log((double)Aj);

    long long A1j;
    if (j % 2 == 0)
        A1j = (Aj * 2 - 1) / 3;
    else
        A1j = (Aj * 4 - 1) / 3;
    int    A1j_mod6 = (int)(A1j % 6);
    double log_A1j  = (A1j > 0) ? log((double)A1j) : 0.0;

    if (i == 0) {
        mod6_mat[0 * n_cols + j] = (signed char)Aj_mod6;
        log_mat [0 * n_cols + j] = log_Aj;
        return;
    }
    if (i == 1) {
        mod6_mat[1 * n_cols + j] = (signed char)A1j_mod6;
        log_mat [1 * n_cols + j] = log_A1j;
        return;
    }

    const double LN2     = 0.6931471805599453;
    const double LOG_2o3 = -0.4054651081081645;  /* log(2/3) */
    const double LOG_4o3 = 0.2876820724517809;   /* log(4/3) */

    int terms    = i - 1;
    int mult_m6  = (j % 2 == 0) ? 2 : 4;
    int G_mod6   = (int)(((long long)terms * Aj_mod6 * mult_m6) % 6);
    int new_mod6 = (A1j_mod6 + G_mod6) % 6;

    double log_Gfactor = (j % 2 == 0) ? LOG_2o3 : LOG_4o3;
    double pow4 = exp(2.0 * (i - 1) * LN2);
    double log_G;
    if (pow4 > 1e15) {
        log_G = log_Aj + log_Gfactor + 2.0 * (i - 1) * LN2;
    } else {
        double G_exact = (j % 2 == 0)
                         ? (2.0 / 3.0) * (pow4 - 1.0)
                         : (4.0 / 3.0) * (pow4 - 1.0);
        log_G = (G_exact > 0) ? log(Aj * G_exact) : log_A1j;
    }

    double log_new;
    double diff = log_A1j - log_G;
    if (diff < -30.0) {
        log_new = log_G;
    } else {
        log_new = log_G + log1p(exp(diff));
    }

    mod6_mat[i * n_cols + j] = (signed char)new_mod6;
    log_mat [i * n_cols + j] = log_new;
}


__global__ void max_log_reduce_kernel(
    const double *data,
    double       *partial_max_val,
    int          *partial_max_idx,
    int           total)
{
    extern __shared__ char smem[];
    double *sval = (double*)smem;
    int    *sidx = (int*)(smem + (blockDim.x / 32) * sizeof(double));

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double val = (gid < total) ? data[gid] : -DBL_MAX;
    int    idx = (gid < total) ? gid        : -1;

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_val = __shfl_down_sync(mask, val, offset);
        int    other_idx = __shfl_down_sync(mask, idx, offset);
        if (other_val > val) { val = other_val; idx = other_idx; }
    }

    int lane   = tid & 31;
    int warpId = tid >> 5;
    if (lane == 0) {
        sval[warpId] = val;
        sidx[warpId] = idx;
    }
    __syncthreads();

    int n_warps = blockDim.x >> 5;
    if (tid < n_warps) {
        val = sval[tid];
        idx = sidx[tid];
    } else {
        val = -DBL_MAX;
        idx = -1;
    }
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            double ov = __shfl_down_sync(mask, val, offset);
            int    oi = __shfl_down_sync(mask, idx, offset);
            if (ov > val) { val = ov; idx = oi; }
        }
    }
    if (tid == 0) {
        partial_max_val[blockIdx.x] = val;
        partial_max_idx[blockIdx.x] = idx;
    }
}


__global__ void count_mod3_kernel(
    const signed char *mod6_mat,
    int               *block_counts,
    int                total)
{
    extern __shared__ int scnt[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    scnt[tid] = (gid < total && mod6_mat[gid] == 3) ? 1 : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scnt[tid] += scnt[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_counts[blockIdx.x] = scnt[0];
}


__global__ void fill_indices_kernel(
    const signed char *mod6_mat,
    const double      *log_mat,
    int               *offsets,      /* exclusive prefix sum of block_counts */
    int               *out_rows,
    int               *out_cols,
    double            *out_log_n,
    int n_cols, int total)
{
    extern __shared__ int sbuf[];
    int *srows  = sbuf;
    int *scols  = sbuf + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int match = (gid < total && mod6_mat[gid] == 3) ? 1 : 0;

    sbuf[tid + blockDim.x * 2] = match;
    __syncthreads();
    int *prefix = sbuf + blockDim.x * 2;
    for (int s = 1; s < (int)blockDim.x; s <<= 1) {
        int val = (tid >= s) ? prefix[tid - s] : 0;
        __syncthreads();
        prefix[tid] += val;
        __syncthreads();
    }
    int local_slot = prefix[tid] - match;

    if (match) {
        int global_slot = offsets[blockIdx.x] + local_slot;
        out_rows  [global_slot] = gid / n_cols;
        out_cols  [global_slot] = gid % n_cols;
        out_log_n [global_slot] = log_mat[gid] - 1.791759469228327; /* log(6) */
    }
}
"""

mod_cuda   = SourceModule(cuda_code, options=["-O3", "--use_fast_math"])
build_kernel   = mod_cuda.get_function("build_matrix_kernel")
reduce_kernel  = mod_cuda.get_function("max_log_reduce_kernel")
count_kernel   = mod_cuda.get_function("count_mod3_kernel")
fill_kernel    = mod_cuda.get_function("fill_indices_kernel")


def _build_A_array(n_cols: int) -> list[int]:

    upper = max(200, n_cols * 6)
    A = []
    v = 3
    while len(A) < n_cols:
        v += 2
        if v % 3 != 0:
            A.append(v)
    return A


def _cpu_exact_value(A: list[int], row: int, col: int) -> int:

    j = col
    Aj = A[j]
    A1j = (Aj * 2 - 1) // 3 if j % 2 == 0 else (Aj * 4 - 1) // 3

    if row == 0:
        return Aj
    if row == 1:
        return A1j

    if j % 2 == 0:
        G = (2 * (4 ** (row - 1) - 1)) // 3
    else:
        G = (4 * (4 ** (row - 1) - 1)) // 3

    return A1j + Aj * G


def build_odd_matrix_gpu(n_rows: int, n_cols: int):
    """
    Launch one thread per cell (i, j).  True O(1) per thread via closed-form.
    Returns GPU buffers and host copies.
    """
    total = n_rows * n_cols

    mod6_gpu = cuda.mem_alloc(total * 1)   # int8
    log_gpu  = cuda.mem_alloc(total * 8)   # float64
    BX, BY = 32, 8
    GX = math.ceil(n_cols / BX)
    GY = math.ceil(n_rows / BY)
    build_kernel(
        mod6_gpu, log_gpu,
        np.int32(n_rows), np.int32(n_cols),
        block=(BX, BY, 1),
        grid =(GX, GY, 1)
    )
    cuda.Context.synchronize()
    mod6_host = np.empty(total, dtype=np.int8)
    log_host  = np.empty(total, dtype=np.float64)
    cuda.memcpy_dtoh(mod6_host, mod6_gpu)
    cuda.memcpy_dtoh(log_host,  log_gpu)

    return mod6_gpu, log_gpu, mod6_host, log_host


def report_max_value_gpu(log_gpu, n_rows: int, n_cols: int, A: list[int]):
    total = n_rows * n_cols
    BLOCK = 512

    def _reduce(data_gpu, size):
        grid = math.ceil(size / BLOCK)
        pval = cuda.mem_alloc(grid * 8)
        pidx = cuda.mem_alloc(grid * 4)
        shared_bytes = (BLOCK // 32) * (8 + 4)
        reduce_kernel(
            data_gpu, pval, pidx,
            np.int32(size),
            block=(BLOCK, 1, 1),
            grid =(grid,  1, 1),
            shared=shared_bytes
        )
        cuda.Context.synchronize()
        return pval, pidx, grid

    pval1, pidx1, g1 = _reduce(log_gpu, total)

    if g1 > 1:
        pval2, pidx2, _ = _reduce(pval1, g1)
        pidx1_h = np.empty(g1, dtype=np.int32)
        pidx2_h = np.empty(1,  dtype=np.int32)
        cuda.memcpy_dtoh(pidx1_h, pidx1)
        cuda.memcpy_dtoh(pidx2_h, pidx2)
        flat_idx = int(pidx1_h[int(pidx2_h[0])])
    else:
        pidx1_h = np.empty(1, dtype=np.int32)
        cuda.memcpy_dtoh(pidx1_h, pidx1)
        flat_idx = int(pidx1_h[0])

    max_row = flat_idx // n_cols
    max_col = flat_idx  % n_cols
    max_val = _cpu_exact_value(A, max_row, max_col)

    k     = math.floor(math.log2(float(max_val)))
    alpha = math.log2(float(max_val))
    mult  = float(max_val) / float(2 ** k)

    print("\n── Final result " + "─" * 40)
    print(f"\n  N ≈ 2^{alpha:.1f}\n")
    print("  Interpretation:\n")
    print("   Not an exact power of 2" if mult != 1.0 else "   Exact power of 2")
    print(f"  Lies between:\n\n      2^{k}  and  2^{k+1}\n")
    print(f"  More precise form (useful for analysis):\n")
    print(f"  N ≈ 2^{k} × {mult:.2f}\n")
    print("─" * 55)

    return max_val, k, alpha, mult


def extract_form_indices_gpu(mod6_gpu, log_gpu, n_rows: int, n_cols: int):

    total = n_rows * n_cols
    BLOCK = 256
    grid  = math.ceil(total / BLOCK)

    block_counts_gpu = cuda.mem_alloc(grid * 4)
    count_kernel(
        mod6_gpu, block_counts_gpu,
        np.int32(total),
        block=(BLOCK, 1, 1),
        grid =(grid,  1, 1),
        shared=BLOCK * 4
    )
    cuda.Context.synchronize()

    block_counts_h = np.empty(grid, dtype=np.int32)
    cuda.memcpy_dtoh(block_counts_h, block_counts_gpu)

    offsets_h = np.concatenate([[0], np.cumsum(block_counts_h[:-1])]).astype(np.int32)
    count     = int(offsets_h[-1] + block_counts_h[-1])

    offsets_gpu = cuda.mem_alloc(grid * 4)
    cuda.memcpy_htod(offsets_gpu, offsets_h)

    out_rows_gpu  = cuda.mem_alloc(count * 4)
    out_cols_gpu  = cuda.mem_alloc(count * 4)
    out_log_n_gpu = cuda.mem_alloc(count * 8)

    fill_kernel(
        mod6_gpu, log_gpu,
        offsets_gpu,
        out_rows_gpu, out_cols_gpu, out_log_n_gpu,
        np.int32(n_cols), np.int32(total),
        block=(BLOCK, 1, 1),
        grid =(grid,  1, 1),
        shared=BLOCK * 4 * 3
    )
    cuda.Context.synchronize()

    rows_h  = np.empty(count, dtype=np.int32)
    cols_h  = np.empty(count, dtype=np.int32)
    log_n_h = np.empty(count, dtype=np.float64)

    if count > 0:
        cuda.memcpy_dtoh(rows_h,  out_rows_gpu)
        cuda.memcpy_dtoh(cols_h,  out_cols_gpu)
        cuda.memcpy_dtoh(log_n_h, out_log_n_gpu)

    order   = np.lexsort((cols_h, rows_h))
    rows_h  = rows_h[order]
    cols_h  = cols_h[order]
    log_n_h = log_n_h[order]

    positions = list(zip(rows_h.tolist(), cols_h.tolist()))
    n_indices = [max(0, int(round(math.exp(ln) - 1))) if ln > 0 else 0
                 for ln in log_n_h.tolist()]

    for (i, j), n in zip(positions, n_indices):
        v = 3 * (2 * n + 1)
    print(f"  ({i:2d}, {j:2d})  value = {v:12d}  n = {n}")

    return positions, log_n_h


def plot_position_map(n_rows, n_cols, positions, log_n_values,
                      save_path="odd_matrix_plot.png"):
    ri   = [p[0] for p in positions]
    ci   = [p[1] for p in positions]
    log_n = np.array(log_n_values, dtype=np.float64)

    fig_w = max(12, n_cols * 0.45)
    fig_h = max(9,  n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sc = ax.scatter(ci, ri, c=log_n, cmap="plasma",
                    s=60, edgecolors="k", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="log(1+n)")

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.invert_yaxis()
    ax.set_title(r"Position map of $3(2n+1)$ elements", fontsize=12)
    ax.set_xlabel("Column $j$", fontsize=11)
    ax.set_ylabel("Row $i$",    fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {save_path}")
    plt.show()
