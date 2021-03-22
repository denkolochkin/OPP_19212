#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <mpi.h>

double t = 0.01;
double eps = 1e-10;

int main(int argc, char **argv) {
    int N = 4;
    int flag = 1;
    double start_time;
    double end_time;
    int process_count;
    int process_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    printf("I'm %d from %d processes\n", process_rank, process_count);
    int number_of_elements = N / process_count;
    if (process_rank == 0) {
         double* A = (double*)malloc(sizeof(double) * N * N);
         double* b = (double*)malloc(sizeof(double) * N);
         double* x = (double*)malloc(sizeof(double) * N);
    }
    double* result = (double*)malloc(sizeof(double) * N);
    double* buffer = (double*)malloc(sizeof(double) * number_of_elements);
    double* part = (double*)malloc(sizeof(double) * N * number_of_elements);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                A[i * N + j] = 1;
            }
            else {
                A[i * N + j] = 2;
            }
        }
        x[i] = 0;
        b[i] = 28;
    }
    double norm_b = 0;
    for (int i = 0; i < number_of_elements; ++i) {
        norm_b += b[i] * b[i];
    }
    norm_b = sqrt(norm_b);
    if (process_rank == 0) {
        start_time = MPI_Wtime();
    }
    MPI_Scatter(A, N * number_of_elements, MPI_DOUBLE, part,
                N * number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    while (flag) {
        double norm = 0;
        for (int i = 0; i < number_of_elements; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++) {
                sum += part[i * N + j] * x[i];
            }
            buffer[i] = sum - b[i];
        }
        for (int i = 0; i < number_of_elements; ++i) {
            norm += buffer[i] * buffer[i];
        }
        norm = sqrt(norm);
        if (norm/norm_b <= eps) {
            flag = 0;
        }
        for (int i = 0; i < number_of_elements; i++) {
            x[i] = x[i] - t * buffer[i];
        }
    }
    MPI_Gather (x, number_of_elements, MPI_DOUBLE, result,
                number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (process_rank == 0) {
        for (int i = 0; i < N; ++i) {
            printf("%f \n", result[i]);
        }
        end_time = MPI_Wtime();
        printf("time taken - %f sec\n", end_time - start_time);
        free(A);
        free(b);
        free(x);
    }
    free(buffer);
    free(part);
    free(result);
    MPI_Finalize();
    return 0;
}
