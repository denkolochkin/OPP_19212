#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <omp.h>

double t = 1e-5;
double eps = 1e-9;

int main() {
    int N = 8192;
    int flag = 1;
    double* A = (double*)malloc(sizeof(double) * N * N);
    double* b = (double*)malloc(sizeof(double) * N);
    double* x = (double*)malloc(sizeof(double) * N);
    double* buffer = (double*)malloc(sizeof(double) * N);
    double* cond = (double*)malloc(sizeof(double) * N);
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
        b[i] = N + 1;
    }
    double norm_b = 0;
    double norm = 0;
    double start_time = omp_get_wtime();
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        #pragma omp for reduction(+:norm_b)
        for (int i = 0; i < N; ++i) {
            norm_b += b[i] * b[i];
        }
        #pragma omp single
        {
            norm_b = sqrt(norm_b);
        }
        while (flag) {
            #pragma omp for reduction(+:norm)
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += A[i * N + j] * x[i];
                }
                buffer[i] = sum - b[i];
                cond[i] = x[i] - t * buffer[i];
                norm += buffer[i] * buffer[i];
            }
            #pragma omp single
            {
                double *tmp = x;
                x = cond;
                cond = tmp;
                norm = sqrt(norm);
                if (norm/norm_b <= eps) {
                    flag = 0;
                }
            }
        }
    }
    double end_time = omp_get_wtime();
    printf("time taken: %f \n", end_time - start_time);
    for (int i = 0; i < N; ++i) {
        printf("%f \n", x[i]);
    }
    free(A);
    free(x);
    free(b);
    free(buffer);
    free(cond);
    return 0;
}