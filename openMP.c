#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <matrix_utils.h>
#include <omp.h>

#define MAX( a, b ) ( ( a > b) ? a : b )
#define INT_N 20000

double scalar_mul(double** left, double** right, int M, int N, double h1, double h2) {
    double sum = 0.0;
    size_t i, j = 1;
    #pragma omp parallel for default(shared) private(i, j) reduction(+:sum) schedule(dynamic)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            sum += left[i][j] * right[i][j];
        }
    }
    return sum * h1 * h2;
}

double** minus(double** from, double **sub, double** sol, int M, int N) {
    size_t i, j = 1;
    #pragma omp parallel for default(shared) private(i, j) schedule(dynamic)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            sol[i][j] = from[i][j] - sub[i][j];
        }
    }
    return sol;
}

double b_bc_ij(double x, double y, double h1, double h2, double eps) {
    double x_cross = 1.5 * (y - 2);

    if (x_cross <= x) {
        return h1;
    } else if (x_cross >= x + h1) {
        return 0.0;
    } else {
        return x + h1 - x_cross;
    }
}

double b_ab_ij(double x, double y, double h1, double h2, double eps) {
    double x_cross = 1.5 * (2 - y);

    if (x_cross >= x + h1) {
        return h1;
    } else if (x_cross <= x) {
        return 0.0;
    } else {
        return x_cross - x;
    }
}

double a_bc_ij(double x, double y, double h1, double h2, double eps) {
    double y_cross = 2 + 2.0 * x / 3.0;

    if (y_cross >= y + h2) {
        return h2;
    } else if (y_cross <= y) {
        return 0;
    } else {
        return y_cross - y;
    }
}

double a_ab_ij(double x, double y, double h1, double h2, double eps) {
    double y_cross = 2 - 2.0 * x / 3.0;

    if (y_cross >= y + h2) {
        return h2;
    } else if (y_cross <= y) {
        return 0;
    } else {
        return y_cross - y;
    }
}

/*
    Fij
*/
double integral(double x, double y, double h1, double h2, double (*f)(double, double, double)) {
    double step = h1 / INT_N;
    double res = 0.0;
    double x_start = x - h1 / 2;
    double y_start = y - h2 / 2;
    #pragma omp parallel for reduction(+: res)
    for (size_t i = 0; i < INT_N; ++i) {
        double cur_res = f(x_start + i * step, y_start, h2);
        res += (cur_res - y_start) * step;
    }
    // printf("%f\n", res);
    return res / (h1 * h2);
}

double ab_temp(double x, double y, double h2) {
    if (x < 0) {
        return y;
    }
    double res = 2.0 - 2.0 * x / 3.0;
    if (y > res) {
        return y;
    }
    if (y + h2 < res) {
        return y + h2;
    }
    return res;
}

double bc_temp(double x, double y, double h2) {
    if (x > 0) {
        return y;
    }
    double res = 2.0 + 2.0 * x / 3.0;
    if (y > res) {
        return y;
    }
    if (y + h2 < res) {
        return y + h2;
    }
    return res;
}

double get_b(double x, double y, double h1, double h2, double eps) {
    double len;
    if (x + h1 <= 0.0) {
        len = b_bc_ij(x, y, h1, h2, eps);
    } else if (x >= 0.0) {
        len = b_ab_ij(x, y, h1, h2, eps);
    } else {
        double bc = b_bc_ij(x, y, -x, h2, eps);
        double ab = b_ab_ij(0, y, h1 + x, h2, eps);
        len = ab + bc;
    }
    return len / h1 + (1 - len / h1) / eps;
}

double get_a(double x, double y, double h1, double h2, double eps) {
    double len;
    if (x <= 0.0) {
        len = a_bc_ij(x, y, h1, h2, eps);
    } else {
        len = a_ab_ij(x, y, h1, h2, eps);
    }

    return len / h2 + (1 - len / h2) / eps;
}

double **get_A_w(double** A_w, double** w, int A1, int A2, int M, int N, double h1, double h2, double eps) {
    size_t i, j = 1;
    #pragma omp parallel for default(shared) private(i, j) schedule(dynamic)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            double x_cur = A1 + i * h1 - h1 / 2;
            double y_cur = A2 + j * h2 - h2 / 2;

            double a_i_next_j = get_a(x_cur + h1, y_cur, h1, h2, eps);
            double a_i_j = get_a(x_cur, y_cur, h1, h2, eps);
            double b_i_j_next = get_b(x_cur, y_cur + h2, h1, h2, eps);
            double b_i_j = get_b(x_cur, y_cur, h1, h2, eps);
            
            A_w[i][j] = - (1.0 / h1) * (a_i_next_j * (w[i + 1][j] - w[i][j]) / h1 - a_i_j * (w[i][j] - w[i - 1][j]) / h1)
                - (1.0 / h2) * (b_i_j_next * (w[i][j + 1] - w[i][j]) / h2 - b_i_j * (w[i][j] - w[i][j - 1]) / h2);
        }
    }
    return A_w;
}

double** init_b(double** B, double h1, double h2, int M, int N, int A1, int A2) {
    size_t i, j = 1;
    #pragma omp parallel for default(shared) private(i, j) schedule(dynamic)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            double ab_t = integral(A1 + i * h1, A2 + j * h2, h1, h2, &ab_temp);
            double bc_t = integral(A1 + i * h1, A2 + j * h2, h1, h2, &bc_temp);
            // printf("%f %f - %f %f\n", A1 + i * h1, A2 + j * h2, ab_t, bc_t);
            B[i][j] = ab_t + bc_t;
        }
    }
    return B;
}


int main(int argc, char **argv) {
    // if (argc != 3) {
    //     printf("Wrong amount of arguments\n");
    //     return 1;
    // }
    const size_t M = atoi(argv[1]);
    const size_t N = atoi(argv[2]);
    const int lim = atoi(argv[3]);

    double A1 = -3.0;
    double B1 = 3.0;
    double A2 = 0.0;
    double B2 = 2.0;

    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;

    double eps = MAX(h1, h2);
    double delta = 1e-6;

    double** w = init_matrix(M, N);
    double** w_next = init_matrix(M, N);
    double** B = init_matrix(M, N);
    double** A_w = init_matrix(M, N);
    double** r = init_matrix(M, N);
    double** A_r = init_matrix(M, N);
    double** tau_r = init_matrix(M, N);

    double **A_B = init_matrix(M, N);


    double tau = 0.0;
    double diff = delta;
    clock_t start = omp_get_wtime();

    B = init_b(B, h1, h2, M, N, A1, A2);
    int count = 0;

    while (diff >= delta) {
        count++;
        cpy_matrix(w_next, w, M, N);
        A_w = get_A_w(A_w, w, A1, A2, M, N, h1, h2, eps);
        r = minus(A_w, B, r, M, N);
        A_r = get_A_w(A_r, r, A1, A2, M, N, h1, h2, eps);
        tau = scalar_mul(r, A_r, M, N, h1, h2) / scalar_mul(A_r, A_r, M, N, h1, h2);
        // printf("tau - %.10f\n", tau);
        tau_r = mul_matrix(r, tau_r, tau, M, N);
        w_next = minus(w, tau_r, w_next, M, N);
        diff = sqrt(scalar_mul(tau_r, tau_r, M, N, h1, h2));
        // printf("%.10f\n", diff);
        if (count < lim) {
            continue;
        } else if (count >= lim && count < lim + 10000) {
            if (count % 2 == 1) {
                printf("%.10f\n", diff);
            }
        } else {
            break;
        }
    }

    // printf("Time - %fsec\n", (omp_get_wtime() - start));

    // printf("(r, r) - %f\n", sqrt(scalar_mul(A_r, r, M, N, h1, h2)));
    // printf("(Ar, Ar) - %f\n", sqrt(scalar_mul(A_r, A_r, M, N, h1, h2)));
    // print_matrix(w, M, N);

    A_B = get_A_w(A_B, B, A1, A2, M, N, h1, h2, eps);

    // printf("(AB, AB) - %f\n", sqrt(scalar_mul(A_B, A_B, M, N, h1, h2)));

    delete_matrix(w, M);
    delete_matrix(w_next, M);
    delete_matrix(B, M);
    delete_matrix(A_w, M);
    delete_matrix(r, M);
    delete_matrix(A_r, M);
    delete_matrix(tau_r, M);
    // printf("Count - %d\n", count);

    return 0;
}