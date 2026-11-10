#ifndef ODDLYTYPED_MATRIX_OPS_H
#define ODDLYTYPED_MATRIX_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Contiguous row-major matrices. Caller frees the result with oddlytyped_free(). */
double *oddlytyped_matmul(const double *a, const double *b,
                   size_t rows_a, size_t cols_a, size_t cols_b);
void oddlytyped_free(void *ptr);

#ifdef __cplusplus
}
#endif
#endif
