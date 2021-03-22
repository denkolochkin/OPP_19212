#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double t = 1e-5;
double eps = 1e-14;

int main(int argc, char **argv) {
    int N = 4096;
    int flag = 1;
    double start_time;
    double end_time;
    int process_count;
    int process_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    double* A = NULL;
    double* b = NULL;
    double* x = NULL;
    printf("I'm %d from %d processes\n", process_rank, process_count);
    int number_of_elements = N / process_count;
    if (process_rank == 0) {
        A = (double*)malloc(sizeof(double) * N * N);
        b = (double*)malloc(sizeof(double) * N);
        x = (double*)malloc(sizeof(double) * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    A[i * N + j] = 1;
                } else {
                    A[i * N + j] = 2;
                }
            }
            b[i] = 8191;
        }
    }
    double* buffer = (double*)malloc(sizeof(double) * number_of_elements);
    double* part_A = (double*)malloc(sizeof(double) * N * number_of_elements);
    double* part_b = (double*)malloc(sizeof(double) * number_of_elements);
    double* part_x = (double*)malloc(sizeof(double) * number_of_elements);
    start_time = MPI_Wtime();
    MPI_Scatter(A,N * number_of_elements, MPI_DOUBLE, part_A,
                N * number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, number_of_elements, MPI_DOUBLE, part_b,
                number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(x, number_of_elements, MPI_DOUBLE, part_x,
                number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    while (flag) {
        double norm = 0;
        double norm_b = 0;
        int k = 0;
        for (int i = 0; i < number_of_elements; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++) {
                sum += part_A[i * N + j] * part_x[k];
                k++;
                if (k == number_of_elements) {
                    MPI_Sendrecv_replace(part_x, number_of_elements, MPI_DOUBLE, (process_rank + 1) % process_count,
                                 123,(process_rank + process_count - 1) % process_count,
                                 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    k = 0;
                }
            }
            buffer[i] = sum - part_b[i];
        }
        for (int i = 0; i < number_of_elements; ++i) {
            norm += buffer[i] * buffer[i];
        }
        double recv_norm;
        MPI_Allreduce(&norm, &recv_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        recv_norm = sqrt(recv_norm);
        for (int i = 0; i < number_of_elements; ++i) {
            norm_b += part_b[i] * part_b[i];
        }
        double recv_norm_b;
        MPI_Allreduce(&norm_b, &recv_norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        recv_norm_b = sqrt(recv_norm_b);
        if (recv_norm/recv_norm_b <= eps) {
            flag = 0;
        }
        for (int i = 0; i < number_of_elements; i++) {
            part_x[i] = part_x[i] - t * buffer[i];
        }
    }
    MPI_Gather(part_x, number_of_elements, MPI_DOUBLE, x,
               number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (process_rank == 0) {
        for (int i = 0; i < N; ++i) {
            printf("%f \n", x[i]);
        }
        end_time = MPI_Wtime();
        printf("time taken - %f sec\n", end_time - start_time);
        free(A);
        free(b);
        free(x);
    }
    free(part_A);
    free(part_b);
    free(part_x);
    MPI_Finalize();
    return 0;
}
