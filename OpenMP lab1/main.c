#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <omp.h>

double t = 1e-4;
double eps = 1e-9;

int main() {
    int N = 8192;
    int flag = 1;
    double* A = (double*)malloc(sizeof(double) * N * N);
    double* b = (double*)malloc(sizeof(double) * N);
    double* x = (double*)malloc(sizeof(double) * N);
    double* buffer = (double*)malloc(sizeof(double) * N);
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
    double start_time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        norm_b += b[i] * b[i];
    }
    norm_b = sqrt(norm_b);
    while (flag) {
        double norm = 0;
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * x[i];
            }
            buffer[i] = sum - b[i];
        }
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            norm += buffer[i] * buffer[i];
        }
        norm = sqrt(norm);
        if (norm/norm_b <= eps) {
            flag = 0;
        }
        double * cond = (double *) malloc(N * sizeof(double));
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
           cond[i] = x[i] - t * buffer[i];
        }
        double *tmp = x;
        x = cond;
        cond = tmp;
        free(cond);
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
    return 0;
}