#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#define nx 50
#define ny 50
#define nz 50
#define a 1

double* func[2];
double* buffer_layer[2];
double fi;
double fj;
double fk;
double eps = 1e-8;
double constant;
int layer0 = 1;
int layer1 = 0;
MPI_Request send_req[2];
MPI_Request recv_req[2];

double phi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double rho(double x, double y, double z) {
    return 6 - a * phi(x, y, z);
}

int calc_edges(int f, int J, int K, double hx, double hy, double hz, int rank,
               int count, const int* per_threads, const int* offsets, double owx, double owy, double owz) {
    for (int j = 1; j < ny; ++j) {
        for (int k = 1; k < nz; ++k) {
            if (rank != 0) {
                int i = 0;
                fi = (func[layer0][(i + 1) * J * K + j * K + k] + buffer_layer[0][j * K + k]) / owx;
                fj = (func[layer0][i * J * K + (j + 1) * K + k] +
                        func[layer0][i * J * K + (j - 1) * K + k]) / owy;
                fk = (func[layer0][i * J * K + j * K + (k + 1)] +
                        func[layer0][i * J * K + j * K + (k - 1)]) / owz;
                func[layer1][i * J * K + j * K + k] =
                        (fi + fj + fk - rho((i + offsets[rank]) * hx, j * hy, k * hz)) / constant;
                if (fabs(func[layer1][i * J * K + j * K + k] -
                         phi((i + offsets[rank]) * hx, j * hy, k * hz)) > eps) {
                    f = 0;
                }
            }
            if (rank != count - 1) {
                int i = per_threads[rank] - 1;
                fi = (buffer_layer[1][j * K + k] + func[layer0][(i - 1) * J * K + j * K + k]) / owx;
                fj = (func[layer0][i * J * K + (j + 1) * K + k] +
                        func[layer0][i * J * K + (j - 1) * K + k]) / owy;
                fk = (func[layer0][i * J * K + j * K + (k + 1)] +
                        func[layer0][i * J * K + j * K + (k - 1)]) / owz;
                func[layer1][i * J * K + j * K + k] =
                        (fi + fj + fk - rho((i + offsets[rank]) * hx, j * hy, k * hz)) / constant;
                if (fabs(func[layer1][i * J * K + j * K + k] -
                         phi((i + offsets[rank]) * hx, j * hy, k * hz)) > eps) {
                    f = 0;
                }
            }
        }
    }
    return f;
}

void fill_layers(const int* per_threads, const int* offsets,
                 int rank, int J, int K, double hx, double hy, double hz) {
    for (int i = 0, start_line = offsets[rank]; i <= per_threads[rank] - 1; i++, start_line++) {
        for (int j = 0; j <= ny; j++) {
            for (int k = 0; k <= nz; k++) {
                if ((start_line != 0) && (j != 0) && (k != 0) && (start_line != nx) && (j != ny) && (k != nz)) {
                    func[0][i * J * K + j * K + k] = 0;
                    func[1][i * J * K + j * K + k] = 0;
                } else {
                    func[0][i * J * K + j * K + k] = phi(start_line * hx, j * hy, k * hz);
                    func[1][i * J * K + j * K + k] = phi(start_line * hx, j * hy, k * hz);
                }
            }
        }
    }
}

void send_data(int J, int K, int rank, int count, const int* per_threads) {
    if (rank != 0) {
        MPI_Isend(&(func[layer0][0]), K * J, MPI_DOUBLE,
                  rank - 1, 0, MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(buffer_layer[0], K * J, MPI_DOUBLE, rank - 1,
                  1, MPI_COMM_WORLD, &recv_req[1]);
    }
    if (rank != count - 1) {
        MPI_Isend(&(func[layer0][(per_threads[rank] - 1) * J * K]), K * J,
                  MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &send_req[1]);
        MPI_Irecv(buffer_layer[1], K * J, MPI_DOUBLE, rank + 1,
                  0, MPI_COMM_WORLD, &recv_req[0]);
    }
}

void receive_data(int rank, int count) {
    if (rank != 0) {
        MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
    }
    if (rank != count - 1) {
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&send_req[1], MPI_STATUS_IGNORE);
    }
}

int calc_center(int f, int J, int K, double hx, double hy, double hz, int rank,
                const int* per_threads, const int* offsets, double owx, double owy, double owz) {
    for (int i = 1; i < per_threads[rank] - 1; ++i) {
        for (int j = 1; j < ny; ++j) {
            for (int k = 1; k < nz; ++k) {
                fi = (func[layer0][(i + 1) * J * K + j * K + k] +
                        func[layer0][(i - 1) * J * K + j * K + k]) / owx;
                fj = (func[layer0][i * J * K + (j + 1) * K + k] +
                        func[layer0][i * J * K + (j - 1) * K + k]) / owy;
                fk = (func[layer0][i * J * K + j * K + (k + 1)] +
                        func[layer0][i * J * K + j * K + (k - 1)]) / owz;
                func[layer1][i * J * K + j * K + k] =
                        (fi + fj + fk - rho((i + offsets[rank]) * hx, j * hy, k * hz)) / constant;
                if (fabs(func[layer1][i * J * K + j * K + k] -
                         phi((i + offsets[rank]) * hx, j * hy, k * hz)) > eps) {
                    f = 0;
                }
            }
        }
    }
    return f;
}

void find_max_diff(int J, int K, double hx, double hy, double hz,
                   int rank, const int* per_threads, const int* offsets) {
    double max = 0;
    double f1;
    for (int i = 1; i < per_threads[rank] - 2; i++) {
        for (int j = 1; j < ny; j++) {
            for (int k = 1; k < nz; k++) {
                if ((f1 = fabs(func[layer1][i * J * K + j * K + k] -
                               phi((i + offsets[rank]) * hx, j * hy, k * hz))) > max) {
                    max = f1;
                }
            }
        }
    }
    double tmp = 0;
    MPI_Allreduce(&max, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        max = tmp;
        printf("max difference: %.9lf\n", max);
    }
}

int main(int argc, char** argv) {
    double dx = 2.0;
    double dy = 2.0;
    double dz = 2.0;
    MPI_Init(&argc, &argv);
    int rank;
    int count;
    int f;
    int tmpF;
    MPI_Comm_size(MPI_COMM_WORLD, &count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int* per_threads = (int*)malloc(sizeof(int) * count);
    int* offsets = (int*)malloc(sizeof(int) * count);
    for (int i = 0, height = nz + 1, tmp = count - (height % count), current_line = 0; i < count; ++i) {
        offsets[i] = current_line;
        per_threads[i] = i < tmp ? (height / count) : (height / count + 1);
        current_line += per_threads[i];
    }
    int I = per_threads[rank];
    int J = (ny + 1);
    int K = (nz + 1);
    func[0] = (double*)malloc(sizeof(double) * I * J * K);
    func[1] = (double*)malloc(sizeof(double) * I * J * K);
    buffer_layer[0] = (double*)malloc(sizeof(double) * J * K);
    buffer_layer[1] = (double*)malloc(sizeof(double) * J * K);
    double hx = dx / nx;
    double hy = dy / ny;
    double hz = dz / nz;
    double hx2 = hx * hx;
    double hy2 = hy * hy;
    double hz2 = hz * hz;
    constant = 2 / hx2 + 2 / hy2 + 2 / hz2 + a;
    fill_layers(per_threads, offsets, rank, J, K, hx, hy, hz);
    double start = MPI_Wtime();
    do {
        f = 1;
        layer0 = 1 - layer0;
        layer1 = 1 - layer1;
        send_data(J, K, rank, count, per_threads);
        f = calc_center(f, J, K, hx, hy, hz, rank, per_threads, offsets, hx2, hy2, hz2);
        receive_data(rank, count);
        f = calc_edges(f, J, K, hx, hy, hz, rank, count, per_threads, offsets, hx2, hy2, hz2);
        MPI_Allreduce(&f, &tmpF, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        f = tmpF;
    } while (f == 0);
    double end = MPI_Wtime();
    if (rank == 0) {
        printf("time taken: %lf\n", end - start);
    }
    find_max_diff(J, K, hx, hy, hz, rank, per_threads, offsets);
    free(buffer_layer[0]);
    free(buffer_layer[1]);
    free(func[0]);
    free(func[1]);
    free(offsets);
    free(per_threads);
    MPI_Finalize();
    return 0;
}
