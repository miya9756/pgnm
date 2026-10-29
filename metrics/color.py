"""Cross-channel products for the color correlation error (numba-compiled).

For each of the four nearest-neighbor lags (right, down, down-right, down-left)
and each ordered pair of different channels (a, b), the mean product of the
standardized residuals w_a(p) * w_b(p + lag), divided by the RMS of w_a and w_b
over the same pixel pairs. Pairs never cross from one block into another.
"""
import numpy as np
from numba import njit, prange

COLOR_LAGS = np.array(((0, 1), (1, 0), (1, 1), (1, -1)), dtype=np.int64)
COLOR_PAIRS = np.array(((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)), dtype=np.int64)


@njit(cache=True, parallel=True)
def color_products(white):
    """Normalized products of different channels at the four COLOR_LAGS."""
    blocks, height, width, _ = white.shape
    output = np.zeros((4, 6), np.float64)
    for index in prange(4):
        dy, dx = COLOR_LAGS[index, 0], COLOR_LAGS[index, 1]
        products = np.zeros(6, np.float64)
        squares_a, squares_b = np.zeros(3, np.float64), np.zeros(3, np.float64)
        count = 0
        for block in range(blocks):
            for row in range(max(0, -dy), min(height, height - dy)):
                for col in range(max(0, -dx), min(width, width - dx)):
                    for c in range(3):
                        a = white[block, row, col, c]
                        b = white[block, row + dy, col + dx, c]
                        squares_a[c] += a * a
                        squares_b[c] += b * b
                    for k in range(6):
                        a, b = COLOR_PAIRS[k, 0], COLOR_PAIRS[k, 1]
                        products[k] += (white[block, row, col, a] *
                                        white[block, row + dy, col + dx, b])
                    count += 1
        if count:
            for k in range(6):
                a, b = COLOR_PAIRS[k, 0], COLOR_PAIRS[k, 1]
                denominator = np.sqrt(squares_a[a] / count) * np.sqrt(squares_b[b] / count)
                output[index, k] = products[k] / count / max(denominator, 1e-12)
    return output
