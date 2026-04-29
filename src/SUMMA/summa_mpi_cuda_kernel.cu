// summa_mpi_cuda.cu - CUDA kernel file
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication: C += A * B
__global__ void matmul_kernel(double *A, double *B, double *C, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < n && col < n) {
    double sum = 0.0;
    for (int k = 0; k < n; k++) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] += sum;
  }
}

// Wrapper function callable from C
extern "C" void cuda_matmul(double *A, double *B, double *C, int n)
{
  double *d_A, *d_B, *d_C;
  size_t bytes = n * n * sizeof(double);
  
  // Allocate device memory
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  
  // Copy to device
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, bytes, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks((n + 15) / 16, (n + 15) / 16);
  matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
  
  // Copy result back
  cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
