//
// Created by blancj4 on 4/18/2026.
//
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <string.h>

#define TILE 16

typedef unsigned long long ticks;
// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void)
{
    unsigned int tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((unsigned long long)tbu0) << 32) | tbl;
}

__global__ void matMul(int *A, int *B, int *C, int n) {
    __shared__ int As[TILE][TILE];
    __shared__ int Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int sum = 0;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {

        // Load A tile
        if (row < n && t * TILE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        // Load B tile
        if (col < n && t * TILE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(int argc, char **argv) {
    int N;

    if (argc < 2) {
        printf("Usage: %s <matrix size>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    int size = N * N * sizeof(int);

    int *A = (int*)malloc(size);
    int *B = (int*)malloc(size);
    int *C = (int*)malloc(size);

    for (int i = 0; i < N*N; i++) {
        A[i] = 2;
        B[i] = 2;
    }

    int *dA, *dB, *dC;

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matMul<<<blocks, threads>>>(dA, dB, dC, N);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix:\n");
    for (int i = 0; i < N*N; i++) {
        printf("%d ", C[i]);
        if ((i+1) % N == 0) printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    return 0;
}