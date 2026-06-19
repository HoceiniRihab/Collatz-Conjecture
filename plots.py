import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# BENCHMARK DATA  (n, time_s)

# parallel
DATA_PARALLEL = [
    ( 10, 0.0008), ( 20, 0.0003), ( 30, 0.0003), ( 40, 0.0004), ( 50, 0.0003),
    ( 60, 0.0003), ( 70, 0.0003), ( 80, 0.0003), ( 90, 0.0003), (100, 0.0003),
    (110, 0.0004), (120, 0.0003), (130, 0.0004), (140, 0.0004), (150, 0.0004),
    (160, 0.0004), (170, 0.0004), (180, 0.0004), (190, 0.0004), (200, 0.0004),
    (210, 0.0004), (220, 0.0004), (230, 0.0005), (240, 0.0005), (250, 0.0005),
    (260, 0.0006), (270, 0.0006), (280, 0.0006), (290, 0.0007), (300, 0.0007),
    (310, 0.0007), (320, 0.0007), (330, 0.0008), (340, 0.0008), (350, 0.0008),
    (360, 0.0009), (370, 0.0009), (380, 0.0009), (390, 0.0011), (400, 0.0011),
    (410, 0.0011), (420, 0.0012), (430, 0.0013), (440, 0.0013), (450, 0.0014),
    (460, 0.0015), (470, 0.0015), (480, 0.0015), (490, 0.0017), (500, 0.0018),
]

# sequential
DATA_SEQUENTIAL = [
    ( 10, 0.0005), ( 20, 0.0013), ( 30, 0.0013), ( 40, 0.0018), ( 50, 0.0018),
    ( 60, 0.0025), ( 70, 0.0035), ( 80, 0.0045), ( 90, 0.0058), (100, 0.0071),
    (110, 0.0087), (120, 0.0116), (130, 0.0133), (140, 0.0180), (150, 0.0167),
    (160, 0.0201), (170, 0.0228), (180, 0.0270), (190, 0.0286), (200, 0.0334),
    (210, 0.0378), (220, 0.0404), (230, 0.0449), (240, 0.0511), (250, 0.0578),
    (260, 0.0609), (270, 0.0789), (280, 0.0726), (290, 0.0793), (300, 0.0845),
    (310, 0.0953), (320, 0.0980), (330, 0.1060), (340, 0.1200), (350, 0.1222),
    (360, 0.1395), (370, 0.1642), (380, 0.1560), (390, 0.1819), (400, 0.1779),
    (410, 0.1873), (420, 0.2207), (430, 0.2047), (440, 0.2192), (450, 0.2768),
    (460, 0.2529), (470, 0.2755), (480, 0.4031), (490, 0.5201), (500, 0.5306),
]


def power_law_fit(ns: np.ndarray, ts: np.ndarray):

    log_n  = np.log(ns)
    log_t  = np.log(ts)
    coeffs = np.polyfit(log_n, log_t, deg=1)
    alpha  = coeffs[0]
    C      = np.exp(coeffs[1])
    t_fit  = C * ns ** alpha

    log_t_pred = np.polyval(coeffs, log_n)
    ss_res     = np.sum((log_t - log_t_pred) ** 2)
    ss_tot     = np.sum((log_t - log_t.mean()) ** 2)
    R2         = 1.0 - ss_res / ss_tot

    return alpha, C, R2, t_fit


def plot_scaling_comparison(
    data_par:  list,
    data_seq:  list,
    save_path: str = "rihab_scaling_comparison.png",
) -> None:

    ns_par = np.array([d[0] for d in data_par], dtype=np.float64)
    ts_par = np.array([d[1] for d in data_par], dtype=np.float64)
    ns_seq = np.array([d[0] for d in data_seq], dtype=np.float64)
    ts_seq = np.array([d[1] for d in data_seq], dtype=np.float64)

    assert np.array_equal(ns_par, ns_seq), "n-grids must match for speedup panel."
    ns = ns_par
    α_par, _, R2_par, fit_par = power_law_fit(ns, ts_par)
    α_seq, _, R2_seq, fit_seq = power_law_fit(ns, ts_seq)
    S = ts_seq / ts_par
    C_PAR  = "#0077b6"   #  parallel
    C_SEQ  = "#d62828"   #sequential
    C_SPD  = "#e07b00"   #speedup
    C_GRID = "#dddddd"
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize     = (9, 8),
        gridspec_kw = {"height_ratios": [3, 1.4], "hspace": 0.08},
        sharex      = True,
    )
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")

    ax1.plot(ns, ts_par, "o-",
             color=C_PAR, lw=1.5, ms=3.5, alpha=0.9,
             label="Parallel (GPU)")
    ax1.plot(ns, fit_par, "--",
             color=C_PAR, lw=1.1, alpha=0.6,
             label=rf"Fit: $t \propto n^{{{α_par:.2f}}}$  ")
    ax1.plot(ns, ts_seq, "s-",
             color=C_SEQ, lw=1.5, ms=3.5, alpha=0.9,
             label="Sequential (CPU)")
    ax1.plot(ns, fit_seq, "--",
             color=C_SEQ, lw=1.1, alpha=0.6,
             label=rf"Fit: $t \propto n^{{{α_seq:.2f}}}$  ")
    ax1.set_ylabel("time  (s)", color="black", fontsize=10)
    ax1.set_title(
        r"Execution Time Scaling",
        color="black", fontsize=12, pad=10,
    )
    ax1.legend(
        fontsize=8.5, framealpha=0.95,
        facecolor="white", edgecolor="#bbbbbb", labelcolor="black",
        loc="upper left",
    )
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x*1e3:.1f} ms" if x < 0.1 else f"{x:.3f} s"
        )
    )
    ax1.grid(True, linestyle="--", alpha=0.5, color=C_GRID)
    ax1.tick_params(axis="y", colors="black")

    for (α, col, xpos, ypos) in [
        (α_seq, C_SEQ, 0.62, 0.88),
        (α_par, C_PAR, 0.62, 0.78),
    ]:
        ax1.annotate(
            rf"$\alpha = {α:.3f}$",
            xy=(xpos, ypos), xycoords="axes fraction",
            color=col, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=col, alpha=0.9),
        )

    ax2.fill_between(ns, 1, S, alpha=0.15, color=C_SPD)
    ax2.plot(ns, S, "o-", color=C_SPD, lw=1.5, ms=3,
             label=r"Speedup  $S = t_\mathrm{seq} / t_\mathrm{par}$")
    ax2.axhline(1, color="#999999", lw=0.8, linestyle="--")

    ax2.set_xlabel(r"Matrix dimension  $n$", color="black", fontsize=10)
    ax2.set_ylabel("Speedup  S(n)", color="black", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.95,
               facecolor="white", edgecolor="#bbbbbb", labelcolor="black")
    ax2.grid(True, linestyle="--", alpha=0.5, color=C_GRID)
    ax2.tick_params(axis="both", colors="black")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"Figure saved → {save_path}")

    print("\n── Scaling summary " + "─" * 44)
    print(f"  GPU  (parallel)   α = {α_par:.3f}   R² = {R2_par:.4f}   "
          f"t(500) = {ts_par[-1]*1e3:.2f} ms")
    print(f"  CPU  (sequential) α = {α_seq:.3f}   R² = {R2_seq:.4f}   "
          f"t(500) = {ts_seq[-1]*1e3:.2f} ms")
    print(f"  Max speedup       S = {S.max():.1f}×  at  n = {int(ns[np.argmax(S)])}")
    print(f"  Mean speedup      S = {S.mean():.1f}×  (n = 10 … 500)")
    print("─" * 63)

    plt.show()


if __name__ == "__main__":
    plot_scaling_comparison(
        data_par  = DATA_PARALLEL,
        data_seq  = DATA_SEQUENTIAL,
        save_path = "rihab_scaling_comparison.png",
    )
