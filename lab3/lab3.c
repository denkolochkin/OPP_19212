#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define x 0
#define y 1

void free_all(double* A, double* B, double* C, double* partA, double* partB, double* partC,
              MPI_Comm* grid_comm, MPI_Comm* col_comm, MPI_Comm* row_comm, int rank) {
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    free(partA);
    free(partB);
    free(partC);
    MPI_Comm_free(grid_comm);
    MPI_Comm_free(col_comm);
    MPI_Comm_free(row_comm);
}

void fill_matrix(double* matrix, int row, int col) {
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j) {
            if (i == j) {
                matrix[i * col + j] = 4;
            }
            else {
                matrix[i * col + j] = 2;
            }
        }
}

void print_matrix(double* matrix, int n1, int n2) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            printf("%lf ", matrix[i * n2 + j]);
        }
        printf("\n");
    }
}

void build_result_matrix(double* partC, double* C, MPI_Comm grid_comm, int dims[2],
                         int coords[2], int n1, int n3, int comm_size) {
    int *recvCounts = (int*)malloc(sizeof(int) * comm_size);
    int *displs = (int*)malloc(sizeof(int) * comm_size);
    MPI_Datatype recv_vector_t, resized_recv_vector_t, send_vector_t;
    PMPI_Type_contiguous(n1 * n3 / (dims[x] * dims[y]), MPI_DOUBLE, &send_vector_t);
    MPI_Type_commit(&send_vector_t);
    MPI_Type_vector(n1 / dims[y], n3 / dims[x], n3, MPI_DOUBLE, &recv_vector_t);
    MPI_Type_commit(&recv_vector_t);
    MPI_Type_create_resized(recv_vector_t, 0, n3 / dims[x] * sizeof(double), &resized_recv_vector_t);
    MPI_Type_commit(&resized_recv_vector_t);
    for (int rank_i = 0; rank_i < comm_size; ++rank_i) {
        recvCounts[rank_i] = 1;
        MPI_Cart_coords(grid_comm, rank_i, 2, coords);
        displs[rank_i] = dims[x] * (n1 / dims[y]) * coords[y] + coords[x];
    }
    MPI_Gatherv(partC, 1, send_vector_t, C, recvCounts, displs, resized_recv_vector_t, 0, grid_comm);
    MPI_Type_free(&recv_vector_t);
    MPI_Type_free(&resized_recv_vector_t);
    MPI_Type_free(&send_vector_t);
}

void init_matrices(double** A, double** B, double** C, int n1, int n2, int n3) {
    *A = (double*)malloc(sizeof(double) * n1 * n2);
    *B = (double*)malloc(sizeof(double) * n2 * n3);
    *C = (double*)malloc(sizeof(double) * n1 * n3);
    fill_matrix(*A, n1, n2);
    fill_matrix(*B, n2, n3);
    printf("A:\n");
    print_matrix(*A, n1, n2);
    printf("B:\n");
    print_matrix(*B, n2, n3);
    printf("A * B:\n");
}

void init_commutators(MPI_Comm* grid_comm, MPI_Comm* rows_comm, MPI_Comm* col_comm, int* coords, int* dims) {
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, grid_comm);
    int rank;
    MPI_Comm_rank(*grid_comm, &rank);
    MPI_Cart_coords(*grid_comm, rank, 2, coords);
    MPI_Comm_split(*grid_comm, coords[y], coords[x], rows_comm);
    MPI_Comm_split(*grid_comm, coords[x], coords[y], col_comm);
}

void distribute_matrices(const double* matrixA, const double* matrixB, double* partA,
                         double* partB, const MPI_Comm row_comm, const MPI_Comm col_comm,
                         const int coords[2], const int dims[2], int n1, int n2, int n3) {
    if (coords[x] == 0) {
        MPI_Scatter(matrixA, n1 * n2 / dims[y], MPI_DOUBLE, partA,
                    n1 * n2 / dims[y], MPI_DOUBLE, 0, col_comm);
    }
    if (coords[y] == 0) {
        MPI_Datatype vector_t;
        MPI_Datatype resized_vector_t;
        MPI_Datatype recv_t;
        MPI_Type_vector(n2, n3 / dims[x], n3, MPI_DOUBLE, &vector_t);
        MPI_Type_commit(&vector_t);
        MPI_Type_create_resized(vector_t, 0, n3 / dims[x] * sizeof(double), &resized_vector_t);
        MPI_Type_commit(&resized_vector_t);
        PMPI_Type_contiguous(n2 * n3 / dims[x], MPI_DOUBLE, &recv_t);
        MPI_Type_commit(&recv_t);
        MPI_Scatter(matrixB, 1, resized_vector_t, partB, 1, recv_t, 0, row_comm);
        MPI_Type_free(&resized_vector_t);
        MPI_Type_free(&vector_t);
        MPI_Type_free(&recv_t);
    }
    MPI_Bcast(partA, n1 * n2 / dims[y], MPI_DOUBLE, 0, row_comm);
    MPI_Bcast(partB, n2 * n3 / dims[x], MPI_DOUBLE, 0, col_comm);
}


void matrix_mul(const double* A, const double* B, double* C, int rowsA, int colA, int colB) {
    for (int i = 0; i < rowsA; ++i) {
        double *c = C + i * colB;
        for (int j = 0; j < colB; ++j) {
            c[j] = 0;
        }
        for (int k = 0; k < colA; ++k) {
            const double *b = B + k * colB;
            double a = A[i * colA + k];
            for (int j = 0; j < colB; ++j) {
                c[j] += a * b[j];
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int dims[2] = {0, 0};
    int coords[2];
    int size;
    int rank;
    double start_time;
    double end_time;
    MPI_Comm grid_comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, 2, dims);
    init_commutators(&grid_comm, &row_comm, &col_comm, coords, dims);
    MPI_Comm_rank(grid_comm, &rank);
    int n1 = 1000;
    int n2 = 1000;
    int n3 = 1000;
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *partA = (double*)malloc(sizeof(double) * n1 * n2 / dims[y]);
    double *partB = (double*)malloc(sizeof(double) * n2 * n3 / dims[x]);
    double *partC = (double*)malloc(sizeof(double) * n1 * n3 / (dims[x] * dims[y]));
    if (rank == 0) {
        start_time = MPI_Wtime();
        init_matrices(&A, &B, &C, n1, n2, n3);
    }
    distribute_matrices(A, B, partA, partB, row_comm, col_comm, coords, dims, n1, n2, n3);
    matrix_mul(partA, partB, partC, n1 / dims[y], n2, n3 / dims[x]);
    build_result_matrix(partC, C, grid_comm, dims, coords, n1, n3, size);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("time taken - %f sec\n", end_time - start_time);
        print_matrix(C, n1, n3);
    }
    free_all(A, B, C, partA, partB, partC, &grid_comm, &col_comm, &row_comm, rank);
    MPI_Finalize();
    return 0;
}
