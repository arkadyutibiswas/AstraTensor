#ifndef ASTRATENSOR_MATRIX_OPS_H
#define ASTRATENSOR_MATRIX_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Contiguous row-major matrices. Caller frees the result with ast_free(). */
double *ast_matmul(const double *a, const double *b,
                   size_t rows_a, size_t cols_a, size_t cols_b);
void ast_free(void *ptr);

#ifdef __cplusplus
}
#endif
#endif
