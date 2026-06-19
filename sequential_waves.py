import math


def build_odd_matrix(n_rows: int, n_cols: int) -> list[list[int]]:

    upper  = max(150, n_cols * 4)
    A_full = [i for i in range(5, upper, 2) if i % 3 != 0]
    A      = A_full[:n_cols]

    A1 = [
        int((A[i] * 2 - 1) / 3) if i % 2 == 0
        else int((A[i] * 4 - 1) / 3)
        for i in range(len(A))
    ]

    rows       = [A, A1]
    a, b, prev = 1, 2, A1

    for _ in range(n_rows):
        M = [
            int(prev[i] + A[i] * (2 ** a)) if i % 2 == 0
            else int(prev[i] + A[i] * (2 ** b))
            for i in range(len(A))
        ]
        rows.append(M)
        a += 2; b += 2; prev = M
        if len(rows) >= n_rows:
            break

    return rows[:n_rows]


def extract_form_indices(matrix: list[list[int]]):

    positions, n_indices = [], []
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            n = (v - 1) // 6
            if v == 3 * (2 * n + 1):
                positions.append((i, j))
                n_indices.append(n)
    return positions, n_indices


def report_max_value(matrix: list[list[int]]):

    max_val = max(max(row) for row in matrix)
    k       = math.floor(math.log2(max_val))
    alpha   = math.log2(float(max_val))
    mult    = max_val / (2 ** k)
    return max_val, k, alpha, mult
