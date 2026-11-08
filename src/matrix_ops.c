#include "matrix_ops.h"
#include <stdint.h>
#include <stdlib.h>

/* i-k-j order reuses A[i,k] and streams through contiguous B[k,:] and C[i,:].
 * The inner loop is unrolled by four, with a scalar tail for remaining cols. */
double *ast_matmul(const double *a, const double *b,
                   size_t rows_a, size_t cols_a, size_t cols_b)
{
    size_t i;
    double *result;

    if (a == NULL || b == NULL || rows_a == 0U ||
        cols_a == 0U || cols_b == 0U) {
        return NULL;
    }
    if (rows_a > SIZE_MAX / cols_b ||
        rows_a * cols_b > SIZE_MAX / sizeof(*result)) {
        return NULL;
    }

    result = (double *)calloc(rows_a * cols_b, sizeof(*result));
    if (result == NULL) {
        return NULL;
    }

    for (i = 0U; i < rows_a; ++i) {
        size_t k;
        double *out_row = result + (i * cols_b);
        for (k = 0U; k < cols_a; ++k) {
            const double a_ik = a[(i * cols_a) + k];
            const double *b_row = b + (k * cols_b);
            size_t j = 0U;
            for (; j + 3U < cols_b; j += 4U) {
                out_row[j] += a_ik * b_row[j];
                out_row[j + 1U] += a_ik * b_row[j + 1U];
                out_row[j + 2U] += a_ik * b_row[j + 2U];
                out_row[j + 3U] += a_ik * b_row[j + 3U];
            }
            for (; j < cols_b; ++j) {
                out_row[j] += a_ik * b_row[j];
            }
        }
    }
    return result;
}

void ast_free(void *ptr)
{
    free(ptr);
}
