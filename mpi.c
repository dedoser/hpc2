#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <matrix_utils.h>
#include <omp.h>

#define MAX( a, b ) ( ( a > b) ? a : b )
#define MIN( a, b ) ( ( a < b) ? a : b )
#define INT_N 20000

struct Task {
  int rank;
  int size;
  // Process topological coords
  int coords[2];
  int left_rank, right_rank, up_rank, down_rank;
  // Process local domain size
  size_t m, n;
  size_t a1, a2, b1, b2;

};
typedef struct Task Task_t;

double scalar_mul(double** left, double** right, int M, int N, double h1, double h2, MPI_Comm *comm) {
    double sum = 0.0;
    size_t i, j = 1;
    double local_sum = 0.0;
    double reduced_sum = 0.0;
    #pragma omp parallel for default(shared) private(i, j) reduction(+:sum) schedule(dynamic)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            local_sum += left[i][j] * right[i][j];
        }
    }
    MPI_Allreduce(&local_sum, &reduced_sum, 1, MPI_DOUBLE, MPI_SUM, *comm);
    return reduced_sum * h1 * h2;
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

double **get_A_w(double** A_w, double** w, int A1, int A2, double h1, double h2, double eps, Task_t *task) {
    size_t i, j = 1;
    size_t M = task->m;
    size_t N = task->n;
    size_t m_start = task->a1;
    size_t n_start = task->b1;
    #pragma omp parallel for default(shared) private(i, j) schedule(dynamic)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            double x_cur = A1 + (m_start + i) * h1 - h1 / 2;
            double y_cur = A2 + (n_start + j) * h2 - h2 / 2;

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

void partitioningDomain(size_t M, size_t N, MPI_Comm *Comm, Task_t* info) {

    int power, px, py;
    int dims[2];
    const int ndims = 2;
    int periods[2] = {0, 0};
    int rank, size;
    // Init MPI lib
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    dims[0] = size;
    dims[1] = 1;
    
    
    int m = M / dims[0]; 
    int n = N / dims[1];
    int rx = M + 1 - dims[0] * m;
    int ry = N + 1 - dims[1] * n;
    
    int coords[2];

    // printf("dims:[%d, %d]", dims[0], dims[1]);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, Comm);
    
    MPI_Cart_coords(*Comm, rank, ndims, coords);
    
    int a1 = MIN(rx, coords[0]) * (m + 1) + MAX(0, (coords[0] - rx)) * m;
    int b1 = MIN(ry, coords[1]) * (n + 1) + MAX(0, (coords[1] - ry)) * n;
    int a2 = a1 + m + (coords[0] < rx ? 1 : 0);
    int b2 = b1 + n + (coords[1] < ry ? 1 : 0);
    m = a2 - a1;
    n = b2 - b1;
    int up, left, down, right;

    MPI_Cart_shift(*Comm, 1, -1, &up, &down);
    MPI_Cart_shift(*Comm, 0, 1, &left, &right);
    info->rank = rank;
    info->coords[0] = coords[0];
    info->coords[1] = coords[1];
    info->left_rank=left;
    info->right_rank=right;
    info->up_rank = up;
    info->down_rank = down;
    info->m = m; 
    info->n = n;
    info->a1 = a1;
    info->a2 = a2;
    info->b1 = b1;
    info->b2 = b2;
    info->size = dims[0] * dims[1];
}

void sendrecv(double **domain, 
              double *send_up_row, double *recv_up_row, 
              double *send_down_row, double *recv_down_row,
              double *send_left_column, double *recv_left_column,
              double *send_right_column, double *recv_right_column,
              MPI_Comm* Comm, Task_t *info) {

    int up = info->up_rank;
    int down = info->down_rank;
    int left = info->left_rank;
    int right = info->right_rank;
    int m = info->m;
    int n = info->n;
    int rank = info->rank;
    int i, j;
    #pragma omp parallel for default(shared) private(i) schedule(dynamic)
    for (i = 0; i < info->m; ++i) {
      send_down_row[i] = domain[i + 1][1];
      send_up_row[i] = domain[i + 1][n]; 
    }
    #pragma omp parallel for default(shared) private(j) schedule(dynamic)
    for (j = 0; j < n; ++j) {  
      send_left_column[j] = domain[1][j + 1]; 
      send_right_column[j] = domain[m][j + 1];
    }
    MPI_Status Status;

	if ((up < 0) && (down >= 0)) {
        
        // printf("Wait1");
		MPI_Sendrecv(send_down_row, m,MPI_DOUBLE, down, 0, recv_down_row, m, MPI_DOUBLE, down, 0, *Comm, &Status);
        // printf("End1");
	}
    else if ((up >= 0) && (down < 0)) {
        // printf("Wait3");
		MPI_Sendrecv(send_up_row, m, MPI_DOUBLE, up, 0, recv_up_row, m, MPI_DOUBLE, up, 0, *Comm, &Status);
        // printf("End3");
	}
	else if ((up >= 0) && (down >= 0)) {
        // printf("Wait2");
		MPI_Sendrecv(send_up_row, m, MPI_DOUBLE, up, 0, recv_up_row, m, MPI_DOUBLE, up, 0, *Comm, &Status);
		MPI_Sendrecv(send_down_row, m, MPI_DOUBLE, down, 0, recv_down_row, m, MPI_DOUBLE, down, 0, *Comm, &Status);
        // printf("End2");
	}

	if ((left < 0) && (right >= 0)) {
		MPI_Sendrecv(send_right_column, n, MPI_DOUBLE, right, 0, recv_right_column, n, MPI_DOUBLE, right, 0, *Comm, &Status);
	}
	else if ((left >= 0) && (right >= 0)) {
		MPI_Sendrecv(send_left_column, n, MPI_DOUBLE, left, 0, recv_left_column, n, MPI_DOUBLE, left, 0, *Comm, &Status);
		MPI_Sendrecv(send_right_column, n, MPI_DOUBLE, right, 0, recv_right_column, n, MPI_DOUBLE, right, 0, *Comm, &Status);

	}
	else if ((left >= 0) && (right < 0)) {
		MPI_Sendrecv(send_left_column, n, MPI_DOUBLE, left, 0, recv_left_column, n, MPI_DOUBLE, left, 0, *Comm, &Status);
	}
    
    // printf("I'm okay %d\n", info->rank);
    #pragma omp parallel for default(shared) private(i) schedule(dynamic)
    for (i = 0; i < m; ++i) {
        domain[i + 1][0] = recv_down_row[i];
        domain[i + 1][n + 1] = recv_up_row[i];
    }
    #pragma omp parallel for default(shared) private(j) schedule(dynamic)
    for (j = 0; j < n; ++j) {
        domain[0][j + 1] = recv_left_column[j];
        domain[m + 1][j + 1] = recv_right_column[j];
    }
}

void solve_eq(MPI_Comm *comm, Task_t *task, double A1, double A2, double h1, double h2, double eps, double delta) {
    int m = task->m;
    int n = task->n;

    double** w = init_matrix(m + 2, n + 2);
    double** w_next = init_matrix(m + 2, n + 2);
    double** B = init_matrix(m + 2, n + 2);
    double** A_w = init_matrix(m + 2, n + 2);
    double** r = init_matrix(m + 2, n + 2);
    double** A_r = init_matrix(m + 2, n + 2);
    double** tau_r = init_matrix(m + 2, n + 2);

    double **A_B = init_matrix(m, n);


    double tau = 0.0;
    double diff_global = delta;
    clock_t start = omp_get_wtime();
    // printf("%d %d %f %f\n", m, n, A1, A2);

    B = init_b(B, h1, h2, m, n, A1, A2);
    int count = 0;

    double *send_up_row =       (double*) malloc(m * sizeof(double));
    double *recv_up_row =       (double*) malloc(m * sizeof(double));
    double *send_down_row =     (double*) malloc(m * sizeof(double));
    double *recv_down_row =     (double*) malloc(m * sizeof(double));
    double *send_left_column =  (double*) malloc(n * sizeof(double));
    double *recv_left_column =  (double*) malloc(n * sizeof(double));
    double *send_right_column = (double*) malloc(n * sizeof(double));
    double *recv_right_column = (double*) malloc(n * sizeof(double));

    while (diff_global >= delta) {
        count++;
        cpy_matrix(w_next, w, m, n);

        sendrecv(w, 
             send_up_row, recv_up_row, 
             send_down_row, recv_down_row, 
             send_left_column, recv_left_column, 
             send_right_column, recv_right_column,
             comm, task);

        A_w = get_A_w(A_w, w, A1, A2, h1, h2, eps, task);
        r = minus(A_w, B, r, m, n);

        sendrecv(w, 
             send_up_row, recv_up_row, 
             send_down_row, recv_down_row, 
             send_left_column, recv_left_column, 
             send_right_column, recv_right_column,
             comm, task);
        // print_matrix(B, m, n);    
        A_r = get_A_w(A_r, r, A1, A2, h1, h2, eps, task);
        tau = scalar_mul(r, A_r, m, n, h1, h2, comm) / scalar_mul(A_r, A_r, m, n, h1, h2, comm);
        tau_r = mul_matrix(r, tau_r, tau, m, n);
        w_next = minus(w, tau_r, w_next, m, n);
        double diff_local = sqrt(scalar_mul(tau_r, tau_r, m, n, h1, h2, comm));
        MPI_Allreduce(&diff_local, &diff_global, 1, MPI_DOUBLE, MPI_MAX, *comm); 

        // printf("%.10f\n", diff);
        // if (count < lim) {
        //     continue;
        // } else if (count >= lim && count < lim + 10000) {
        //     if (count % 2 == 1) {
        //         printf("%.10f\n", diff);
        //     }
        // } else {
        //     break;
        // }
        printf("diff_glob - %.10f\n", diff_global);

    }

    // printf("Time - %fsec\n", (omp_get_wtime() - start));

    // printf("(r, r) - %f\n", sqrt(scalar_mul(A_r, r, M, N, h1, h2)));
    // printf("(Ar, Ar) - %f\n", sqrt(scalar_mul(A_r, A_r, M, N, h1, h2)));
    // print_matrix(w, M, N);
    // printf("(AB, AB) - %f\n", sqrt(scalar_mul(A_B, A_B, M, N, h1, h2)));

    delete_matrix(w, m);
    delete_matrix(w_next, m);
    delete_matrix(B, m);
    delete_matrix(A_w, m);
    delete_matrix(r, m);
    delete_matrix(A_r, m);
    delete_matrix(tau_r, m);
    // printf("Count - %d\n", count);

}


int main(int argc, char **argv) {
    // if (argc != 3) {
    //     printf("Wrong amount of arguments\n");
    //     return 1;
    // }
    const size_t M = atoi(argv[1]);
    const size_t N = atoi(argv[2]);
    // const int lim = atoi(argv[3]);

    double A1 = -3.0;
    double B1 = 3.0;
    double A2 = 0.0;
    double B2 = 2.0;

    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;

    double eps = MAX(h1, h2);
    double delta = 1e-6;

    Task_t task;
    MPI_Comm comm;
    MPI_Init(&argc, &argv);

    partitioningDomain(M, N, &comm, &task);
    // printf("%d %d %d\n", task.rank, task.m, task.n);
    solve_eq(&comm, &task, A1, A2, h1, h2, eps, delta);
    MPI_Finalize();
    return 0;
}