#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>


extern "C" {
    // perform multiplication of mat_A and mat_B into mat_C
    // these will be sub matrices that each rank contains
    void launchMultiplyMatrices(int* mat_A, int* mat_B, int* mat_C, int dimension, int myrank);

    // allocates a chunk of memory that can be used on both the GPUs and the CPUs
    int* allocate_memory(int size);

    // free up managed memory
    void free_memory(int* ptr);
}


__global__
void multiplyMatrices(int* mat_A, int* mat_B, int* mat_C, int dimension) {
    // temp should be the 2 * sizeof(mat_A). It will store mat_A and mat_B
    // extern __shared__ temp[]

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < dimension && col < dimension) {
        int sum = 0;
        for (int i = 0; i < dimension; i++) {
            sum += (mat_A[row * dimension + i] * mat_B[dimension * i + col]);
        }

        mat_C[row * dimension + col] += sum;
    }
}


void launchMultiplyMatrices(int* mat_A, int* mat_B, int* mat_C, int dimension, int myrank) {
    int devID = myrank % 4;
    cudaSetDevice(devID);

    cudaMemPrefetchAsync(mat_A, dimension*dimension*sizeof(int), devID);
    cudaMemPrefetchAsync(mat_B, dimension*dimension*sizeof(int), devID);
    cudaMemPrefetchAsync(mat_C, dimension*dimension*sizeof(int), devID);

    dim3 blockSize(16, 16);
    dim3 gridSize( (dimension + blockSize.x - 1) / blockSize.x, (dimension + blockSize.y - 1) / blockSize.y );
    multiplyMatrices<<<gridSize, blockSize>>>(mat_A, mat_B, mat_C, dimension);
    
    cudaDeviceSynchronize();
}


// allocates a chunk of memory that can be used on both the GPUs and the CPUs
int* allocate_memory(int size) {
    int* ptr;
    cudaMallocManaged(&ptr, size*sizeof(int));
    return ptr;
}


// free up managed memory
void free_memory(int* ptr) {
    cudaFree(ptr);
}