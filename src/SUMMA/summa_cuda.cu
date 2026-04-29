// summa_cuda.cu - Pure CUDA implementation (single GPU)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef unsigned long long ticks;

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

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(double *A, double *B, double *C, int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < N && col < N) {
    double sum = 0.0;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <matrix_size>\n", argv[0]);
    return 1;
  }
  
  int N = atoi(argv[1]);
  
  printf("=== SUMMA CUDA-Only Matrix Multiplication ===\n");
  printf("Matrix size:      %d x %d\n", N, N);
  printf("Single GPU\n");
  printf("\n");
  
  size_t bytes = N * N * sizeof(double);
  
  // Allocate host memory
  double *h_A = (double*)malloc(bytes);
  double *h_B = (double*)malloc(bytes);
  double *h_C = (double*)malloc(bytes);
  
  // Initialize matrices
  for (int i = 0; i < N * N; i++) {
    h_A[i] = 1.0;
    h_B[i] = 1.0;
  }
  
  // Allocate device memory
  double *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  
  // Copy to device
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks((N + 15) / 16, (N + 15) / 16);
  
  ticks start = getticks();
  
  matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
  
  ticks end = getticks();
  
  // Copy result back
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
  
  double time_sec = (double)(end - start) / 512000000.0;
  double gflops = (2.0 * N * N * N) / (time_sec * 1e9);
  
  printf("Time:     %.6f seconds\n", time_sec);
  printf("GFLOPS:   %.3f\n", gflops);
  printf("Result check: C[0,0] = %.1f (expected: %.1f)\n", h_C[0], (double)N);
  
  // Free memory
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
