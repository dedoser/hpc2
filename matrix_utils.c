#include <matrix_utils.h>

/**
  Инициализация нулевой матрицы размера (M + 1, N + 1)
*/
double** init_matrix(int M, int N) {
    double** res = (double**) calloc((M + 1), sizeof(double*));
    for (size_t i = 0; i <= M; ++i) {
        res[i] = (double*) calloc((N + 1), sizeof(double));
    }
    return res;
}

void delete_matrix(double** matrix, int M) {
    for (size_t i = 0; i <= M; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

void print_matrix(double** matrix, int M, int N) {
    for (size_t i = 0; i <= M; ++i) {
        for (size_t j = 0; j <= N; ++j) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void cpy_matrix(double **from, double **to, int M, int N) {
    for (size_t i = 0; i <= M; ++i) {
        for (size_t j = 0; j <= N; ++j) {
            to[i][j] = from[i][j];
        }
    }
}

double **mul_matrix(double** matrix, double** orig, double mul, int M, int N) {
    for (size_t i = 0; i <= M; ++i) {
        for (size_t j = 0; j <= N; ++j) {
            orig[i][j] = matrix[i][j] * mul;
        }
    }
    return orig;
}

