// summa_mpi_cuda.c - MPI wrapper
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// External CUDA function
extern void cuda_matmul(double *A, double *B, double *C, int n);

// SUMMA with CUDA acceleration
void summa_mpi_cuda(double *A_local, double *B_local, double *C_local, 
                    int n, int block_size, MPI_Comm grid_comm, int rank)
{
  int grid_rank, grid_size;
  MPI_Comm_rank(grid_comm, &grid_rank);
  MPI_Comm_size(grid_comm, &grid_size);
  
  int dims[2], periods[2], coords[2];
  dims[0] = dims[1] = (int)sqrt(grid_size);
  periods[0] = periods[1] = 0;
  
  MPI_Cart_coords(grid_comm, grid_rank, 2, coords);
  
  MPI_Comm row_comm, col_comm;
  int remain_dims[2];
  
  remain_dims[0] = 0; remain_dims[1] = 1;
  MPI_Cart_sub(grid_comm, remain_dims, &row_comm);
  
  remain_dims[0] = 1; remain_dims[1] = 0;
  MPI_Cart_sub(grid_comm, remain_dims, &col_comm);
  
  double *A_buffer = (double*)malloc(block_size * block_size * sizeof(double));
  double *B_buffer = (double*)malloc(block_size * block_size * sizeof(double));
  
  // Set GPU based on rank
  int gpu_id = rank % 4;  // Assumes 4 GPUs per node
  cudaSetDevice(gpu_id);
  
  for (int k = 0; k < dims[0]; k++) {
    if (coords[1] == k) {
      memcpy(A_buffer, A_local, block_size * block_size * sizeof(double));
    }
    MPI_Bcast(A_buffer, block_size * block_size, MPI_DOUBLE, k, row_comm);
    
    if (coords[0] == k) {
      memcpy(B_buffer, B_local, block_size * block_size * sizeof(double));
    }
    MPI_Bcast(B_buffer, block_size * block_size, MPI_DOUBLE, k, col_comm);
    
    // Use CUDA for local multiply
    cuda_matmul(A_buffer, B_buffer, C_local, block_size);
  }
  
  free(A_buffer);
  free(B_buffer);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
}

int main(int argc, char *argv[])
{
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (argc != 2) {
    if (rank == 0) {
      printf("Usage: %s <matrix_size>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }
  
  int N = atoi(argv[1]);
  int p = (int)sqrt(size);
  
  if (p * p != size || N % p != 0) {
    if (rank == 0) {
      printf("ERROR: Invalid configuration\n");
    }
    MPI_Finalize();
    return 1;
  }
  
  int block_size = N / p;
  
  if (rank == 0) {
    printf("=== SUMMA MPI+CUDA Matrix Multiplication ===\n");
    printf("Matrix size:      %d x %d\n", N, N);
    printf("MPI ranks:        %d (%dx%d grid)\n", size, p, p);
    printf("Block size:       %d x %d\n", block_size, block_size);
    printf("\n");
  }
  
  int dims[2] = {p, p};
  int periods[2] = {0, 0};
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
  
  double *A_local = (double*)malloc(block_size * block_size * sizeof(double));
  double *B_local = (double*)malloc(block_size * block_size * sizeof(double));
  double *C_local = (double*)calloc(block_size * block_size, sizeof(double));
  
  for (int i = 0; i < block_size * block_size; i++) {
    A_local[i] = 1.0;
    B_local[i] = 1.0;
  }
  
  MPI_Barrier(grid_comm);
  ticks start = getticks();
  
  summa_mpi_cuda(A_local, B_local, C_local, N, block_size, grid_comm, rank);
  
  MPI_Barrier(grid_comm);
  ticks end = getticks();
  
  if (rank == 0) {
    double time_sec = (double)(end - start) / 512000000.0;
    double gflops = (2.0 * N * N * N) / (time_sec * 1e9);
    printf("Time:     %.6f seconds\n", time_sec);
    printf("GFLOPS:   %.3f\n", gflops);
    printf("Result check: C[0,0] = %.1f (expected: %.1f)\n", 
           C_local[0], (double)N);
  }
  
  free(A_local);
  free(B_local);
  free(C_local);
  MPI_Comm_free(&grid_comm);
  MPI_Finalize();
  return 0;
}
