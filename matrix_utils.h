#include <stdlib.h>
#include <stdio.h>

double** init_matrix(int M, int N);

void delete_matrix(double** matrix, int M);

void print_matrix(double** matrix, int M, int N);

void cpy_matrix(double **from, double **to, int M, int N);

double **mul_matrix(double** matrix, double** orig, double mul, int M, int N);