//
// Created by blancj4 on 4/26/2026.
//
#include <cuda_runtime.h>

__global__ void blockMulKernel(int *A, int *B, int *C, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    for (int k = 0; k < n; k++) {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] += sum;
}

extern "C" void gpu_block_mul(int *A, int *B, int *C, int n) {

    int size = n * n * sizeof(int);

    int *dA, *dB, *dC;

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);

    dim3 threads(16,16);

    blockMulKernel<<<1, threads>>>(dA, dB, dC, n);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}