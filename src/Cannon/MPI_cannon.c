//
// Created by blancj4 on 4/26/2026.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void gpu_block_mul(int *A, int *B, int *C, int n);

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

//#define N 4

int main(int argc, char** argv) {
    ticks start, end;
    int rank, size;
    int N;

    if (argc < 2) {
        printf("Usage: %s <matrix size>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    int *A_local = malloc(blockSize * sizeof(int));
    int *B_local = malloc(blockSize * sizeof(int));
    int *C_local = calloc(blockSize, sizeof(int));

    if (rank == 0) {
        A = malloc(N*N*sizeof(int));
        B = malloc(N*N*sizeof(int));
        C = malloc(N*N*sizeof(int));

        for (int i = 0; i < N*N; i++) {
            A[i] = 2;
            B[i] = 2;
        }
    }

    // Create block datatype
    MPI_Datatype blockType;
    MPI_Type_vector(block, block, N, MPI_INT, &blockType);
    MPI_Type_create_resized(blockType, 0, sizeof(int), &blockType);
    MPI_Type_commit(&blockType);

    int *sendcounts = NULL, *displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(size*sizeof(int));
        displs = malloc(size*sizeof(int));

        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                sendcounts[i*q + j] = 1;
                displs[i*q + j] = i*N*block + j*block;
            }
        }
        start = getticks();
    }


    // Scatter blocks
    MPI_Scatterv(A, sendcounts, displs, blockType,
                 A_local, blockSize, MPI_INT, 0, comm2d);

    MPI_Scatterv(B, sendcounts, displs, blockType,
                 B_local, blockSize, MPI_INT, 0, comm2d);

    // Setup neighbors
    int left, right, up, down;

    MPI_Cart_shift(comm2d, 1, -1, &right, &left);
    MPI_Cart_shift(comm2d, 0, -1, &down, &up);

    // Initial skew
    for (int i = 0; i < row; i++) {
        MPI_Sendrecv_replace(A_local, blockSize, MPI_INT,
                             left, 0, right, 0, comm2d, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < col; i++) {
        MPI_Sendrecv_replace(B_local, blockSize, MPI_INT,
                             up, 0, down, 0, comm2d, MPI_STATUS_IGNORE);
    }

    // Cannon loop
    for (int step = 0; step < q; step++) {

        // GPU compute
        gpu_block_mul(A_local, B_local, C_local, block);

        // Shift A left
        MPI_Sendrecv_replace(A_local, blockSize, MPI_INT,
                             left, 0, right, 0, comm2d, MPI_STATUS_IGNORE);

        // Shift B up
        MPI_Sendrecv_replace(B_local, blockSize, MPI_INT,
                             up, 0, down, 0, comm2d, MPI_STATUS_IGNORE);
    }

    // Gather result
    MPI_Gatherv(C_local, blockSize, MPI_INT,
                C, sendcounts, displs, blockType,
                0, comm2d);

    if (rank == 0) {
        end = getticks();
        //printf("Result Matrix:\n");
        //for (int i = 0; i < N*N; i++) {
        //    printf("%d ", C[i]);
        //    if ((i+1)%N==0) printf("\n");
        //}
        printf("Total time in MPI/Cuda process is %lf seconds \n", (double)(end - start) / (double)512000000.0);
    }

    free(A_local);
    free(B_local);
    free(C_local);

    if (rank == 0) {
        free(A); free(B); free(C);
        free(sendcounts); free(displs);
    }

    MPI_Type_free(&blockType);

    MPI_Finalize();
    return 0;
}