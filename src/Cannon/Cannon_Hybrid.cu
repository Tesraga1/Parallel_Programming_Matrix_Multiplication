//
// Created by blancj4 on 4/26/2026.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

/*
#define N 4

__global__ void matMulKernel(int *A, int *B, int *C, int rows, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = N / size;

    int *A = NULL, *B = NULL, *C = NULL;
    int *A_local = malloc(rows * N * sizeof(int));
    int *C_local = malloc(rows * N * sizeof(int));

    if (rank == 0) {
        A = malloc(N * N * sizeof(int));
        B = malloc(N * N * sizeof(int));
        C = malloc(N * N * sizeof(int));

        for (int i = 0; i < N*N; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    } else {
        B = malloc(N * N * sizeof(int));
    }

    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows * N, MPI_INT,
                A_local, rows * N, MPI_INT,
                0, MPI_COMM_WORLD);

    // GPU setup
    int *dA, *dB, *dC;

    cudaMalloc(&dA, rows * N * sizeof(int));
    cudaMalloc(&dB, N * N * sizeof(int));
    cudaMalloc(&dC, rows * N * sizeof(int));

    cudaMemcpy(dA, A_local, rows * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (rows + 15)/16);

    matMulKernel<<<blocks, threads>>>(dA, dB, dC, rows, N);

    cudaMemcpy(C_local, dC, rows * N * sizeof(int), cudaMemcpyDeviceToHost);

    MPI_Gather(C_local, rows * N, MPI_INT,
               C, rows * N, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result Matrix:\n");
        for (int i = 0; i < N*N; i++) {
            printf("%d ", C[i]);
            if ((i+1)%N==0) printf("\n");
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A_local);
    free(C_local);
    free(B);

    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
} */

#define N 8   // must be divisible by sqrt(p)

void init_matrix(int *M, int n) {
    for (int i = 0; i < n*n; i++)
        M[i] = 2;
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // GPU assignment
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % deviceCount);

    // 2D grid
    int q = (int)sqrt(size);
    if (q*q != size) {
        if (rank == 0) printf("Processes must be perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Comm comm2d;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2d);

    int coords[2];
    MPI_Cart_coords(comm2d, rank, 2, coords);

    int row = coords[0];
    int col = coords[1];

    int block = N / q;
    int blockSize = block * block;

    int *A = NULL, *B = NULL, *C = NULL;

    //Allocate and initialize matrices
    if (rank == 0) {
        A = (int*)malloc(N*N*sizeof(int));
        B = (int*)malloc(N*N*sizeof(int));
        C = (int*)malloc(N*N*sizeof(int));

        init_matrix(A, N);
        init_matrix(B, N);
    }

    // Allocate GPU memory
    int *dA, *dB, *dC;
    cudaMalloc(&dA, blockSize*sizeof(int));
    cudaMalloc(&dB, blockSize*sizeof(int));
    cudaMalloc(&dC, blockSize*sizeof(int));
    cudaMemset(dC, 0, blockSize*sizeof(int));

    // Create MPI datatype for blocks
    MPI_Datatype blockType;
    MPI_Type_vector(block, block, N, MPI_INT, &blockType);
    MPI_Type_create_resized(blockType, 0, sizeof(int), &blockType);
    MPI_Type_commit(&blockType);

    int *sendcounts = NULL, *displs = NULL;

    if (rank == 0) {
        sendcounts = (int*)malloc(size*sizeof(int));
        displs = (int*)malloc(size*sizeof(int));

        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                sendcounts[i*q + j] = 1;
                displs[i*q + j] = i*N*block + j*block;
            }
        }
    }

    // Scatter into GPU memory
    MPI_Scatterv(A, sendcounts, displs, blockType,
                 dA, blockSize, MPI_INT, 0, comm2d);

    MPI_Scatterv(B, sendcounts, displs, blockType,
                 dB, blockSize, MPI_INT, 0, comm2d);

    // Neighbors
    int left, right, up, down;
    MPI_Cart_shift(comm2d, 1, -1, &right, &left);
    MPI_Cart_shift(comm2d, 0, -1, &down, &up);

    // Initial skew
    for (int i = 0; i < row; i++) {
        MPI_Sendrecv_replace(dA, blockSize, MPI_INT,
                             left, 0, right, 0, comm2d, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < col; i++) {
        MPI_Sendrecv_replace(dB, blockSize, MPI_INT,
                             up, 0, down, 0, comm2d, MPI_STATUS_IGNORE);
    }

    dim3 threads(block, block);

    // Cannon loop
    for (int step = 0; step < q; step++) {

        matMulTiled<<<1, threads>>>(dA, dB, dC, block);
        cudaDeviceSynchronize();

        // Shift A left
        MPI_Sendrecv_replace(dA, blockSize, MPI_INT,
                             left, 0, right, 0, comm2d, MPI_STATUS_IGNORE);

        // Shift B up
        MPI_Sendrecv_replace(dB, blockSize, MPI_INT,
                             up, 0, down, 0, comm2d, MPI_STATUS_IGNORE);
    }

    // Gather result
    MPI_Gatherv(dC, blockSize, MPI_INT,
                C, sendcounts, displs, blockType,
                0, comm2d);

    if (rank == 0) {
        printf("Final Result Matrix:\n");
        for (int i = 0; i < N*N; i++) {
            printf("%d ", C[i]);
            if ((i+1)%N==0) printf("\n");
        }
    }

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    if (rank == 0) {
        free(A); free(B); free(C);
        free(sendcounts); free(displs);
    }

    MPI_Type_free(&blockType);

    MPI_Finalize();
    return 0;
}