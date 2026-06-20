# The Wave-Like Collatz Sequence

Mini research project module, by **Rihab Hoceini**, under the supervision of **Prof. Hofmann**.

## Contribution

This work introduces and studies a structured pattern of sub-series found inside the
Collatz sequence, sitting in the middle of its apparent randomness — elements of the
form `3(2n+1)` recur at predictable positions inside an `n × m` "odd matrix" built from
the sequence's odd-step dynamics. We hope this structured sub-pattern can eventually
contribute to a full formal proof of the Collatz conjecture for all numbers.

## Scaling

The construction and extraction were scaled to large `n × n` matrices using GPU-parallel
computing (CUDA via PyCUDA), benchmarked against the sequential CPU version. Our scaling
limit was **n = 600**, beyond which the available GPU memory/runtime could no longer
sustain the computation.

If you have a good GPU lying around, give it a try for bigger `n` and let us know the
maximum value you managed to reach — we promise to be only mildly jealous.

## Repository contents

- `sequential_wave.py` — CPU (sequential) construction of the odd matrix and extraction of the `3(2n+1)` sub-series.
- `parallel_wave.py` — GPU (CUDA/PyCUDA) parallel construction and extraction, one thread per matrix cell.
- `plot_scaling.py` — scalability comparison plot (sequential vs. parallel) with power-law fit and speedup curve.
- `main.py` — entry point; run either mode with custom matrix dimensions `n` (rows) and `m` (columns).

## Usage

```bash
python main.py --mode sequential --n 11 --m 25
python main.py --mode parallel --n 500 --m 500
```
