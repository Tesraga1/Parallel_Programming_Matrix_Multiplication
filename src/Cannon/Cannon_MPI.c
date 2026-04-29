//
// Created by blancj4 on 4/18/2026.
//
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include <mpi.h>

extern void runCuda();

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

//Matrix structure:
// (4,2) Columns
//  Rows [2,2]
//       [2,2]
//       [2,2]
//       [2,2]

void allocate_matrix(int** matrix, int rows, int columns) {
    int i, j;
    for (i = 0; i < rows; i++)
        matrix[i] = (int*)malloc(columns * sizeof(int));
    for (i = 0; i < rows; i++)
        for (j = 0; j < columns; j++)
            matrix[i][j] = 2;

}

int* get_columns(int** matrix, int rows, int c){
    int* column = (int*)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++){
        column[i] = matrix[i][c];
    }
    return column;
}

void free_matrix(int**matrix, int r){
    for (int i = 0; i < r; i++)
        free(matrix[i]);
    free(matrix);
}

void print_matrix(int** matrix, int rows, int columns){
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}


// Multiply the row and column given for a matrix
// Row_num: number of items in the row
// Column_num: number of items in the column
int multiply_rows_columns(int row_num, int column_num, const int* row, const int* column){
    int p = 0;
    if (row_num != column_num){
        printf("Cannot be multiplied together");
        exit(1);
    }
    for (int i = 0; i < row_num; i++) {
        p += row[i] * column[i];
    }
    return p;
}


/* Normal Main for inital testing
int main(int argv, char** argc){

    int rows_a = 4;//atoi(argc[1]);
    int columns_a = 20;//atoi(argc[2]);
    int rows_b = 20;//atoi(argc[3]);
    int columns_b = 15;//atoi(argc[4]);

    //Allocate Matrices
    int** matrix_a = (int**)malloc(rows_a * sizeof(int*));
    int** matrix_b = (int**)malloc(rows_b * sizeof(int*));
    int** matrix_c = (int**)malloc(rows_a * sizeof(int*));
    allocate_matrix(matrix_a, rows_a,columns_a);
    allocate_matrix(matrix_b, rows_b, columns_b);
    allocate_matrix(matrix_c, rows_a, columns_b);

    print_matrix(matrix_a, rows_a, columns_a);
    print_matrix(matrix_b, rows_b, columns_b);

    //Main algorithm
    for (int i = 0; i < rows_a; i++){
        for (int y = 0; y < columns_b; y++) {
            int *c = get_columns(matrix_b, rows_b, y);
            int p = multiply_rows_columns(columns_a, rows_b, matrix_a[i], c);
            matrix_c[i][y] = p;
            free(c);
        }
    }
    print_matrix(matrix_c, rows_a, columns_b);


    free_matrix(matrix_a,rows_a);
    free_matrix(matrix_b,rows_b);
    free_matrix(matrix_c, rows_a);

    return 0;
}
*/

/*
int main(int argc, char** argv){
    int size, rank, i, j, k, shift_source, shift_dest;
    int ** A = (int**)malloc(N * sizeof(int*));
    int ** B = (int**)malloc(N * sizeof(int*));
    int ** C = (int**)malloc(N * sizeof(int*));
    int ** A_local, B_local, C_local;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0){
        printf("Number of precesses: %d\n", size);
        allocate_matrix(A, N, N/size);
        allocate_matrix(B, N/size, N);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Broadcast the matrices to all processes
    MPI_Bcast(A, N*N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);
    // Scatter matrix A to all processes
    MPI_Scatter(A, N*N/size, MPI_INT, A_local, N*N/size, MPI_INT, 0, MPI_COMM_WORLD);
    // Shift the columns of B by process rank
    MPI_Cart_create(MPI_COMM_WORLD, 1, &size, &rank, 1, &comm);
    MPI_Cart_shift(comm, 1, -rank, &shift_source, &shift_dest);
    MPI_Sendrecv_replace(B, N*N/size, MPI_INT, shift_dest, 0, shift_source, 0, comm, &status);
    // Perform local matrix multiplication
    for (i = 0; i < N/size; i++) {
        for (j = 0; j < N; j++) {
            C_local[i][j] = 0;
            for (k = 0; k < N; k++) {
                C_local[i][j] += A_local[i][k] * B[k][j];
            }
        }
    }


    MPI_Finalize();
} */


//#define N 4   // Matrix size (must be divisible by sqrt(p))

int main(int argc, char *argv[]) {
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
    if (q * q != size) {
        if (rank == 0)
            printf("Number of processes must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Comm comm_2d;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

    int coords[2];
    MPI_Cart_coords(comm_2d, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    int block = N / q;

    int *A = NULL, *B = NULL, *C = NULL;

    int *A_local = malloc(block * block * sizeof(int));
    int *B_local = malloc(block * block * sizeof(int));
    int *C_local = malloc(block * block * sizeof(int));

    if (rank == 0) {
        A = malloc(N * N * sizeof(int));
        B = malloc(N * N * sizeof(int));
        C = malloc(N * N * sizeof(int));

        //printf("Matrix A:\n");
        for (int i = 0; i < N * N; i++) {
            A[i] = 2;
            //printf("%d ", A[i]);
            //if ((i + 1) % N == 0) printf("\n");
        }

        //printf("Matrix B:\n");
        for (int i = 0; i < N * N; i++) {
            B[i] = 2;
            //printf("%d ", B[i]);
            //if ((i + 1) % N == 0) printf("\n");
        }
    }

    // Create datatype for block
    MPI_Datatype block_type;
    MPI_Type_vector(block, block, N, MPI_INT, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(int), &block_type);
    MPI_Type_commit(&block_type);

    int *sendcounts = NULL;
    long long *displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                sendcounts[i * q + j] = 1;
                displs[i*q + j] = (i * block) * N + (j * block);
            }
        }
        start = getticks();
    }

    // Scatter A and B
    MPI_Scatterv(A, sendcounts, displs, block_type,
                 A_local, block * block, MPI_INT, 0, comm_2d);

    MPI_Scatterv(B, sendcounts, displs, block_type,
                 B_local, block * block, MPI_INT, 0, comm_2d);

    // Setup neighbors
    int left, right, up, down;

    MPI_Cart_shift(comm_2d, 1, -1, &right, &left);
    MPI_Cart_shift(comm_2d, 0, -1, &down, &up);

    // Initial skew
    for (int i = 0; i < row; i++) {
        MPI_Sendrecv_replace(A_local, block * block, MPI_INT,
                             left, 0, right, 0, comm_2d, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < col; i++) {
        MPI_Sendrecv_replace(B_local, block * block, MPI_INT,
                             up, 0, down, 0, comm_2d, MPI_STATUS_IGNORE);
    }

    // Main loop
    for (int step = 0; step < q; step++) {

        // Multiply
        for (int i = 0; i < block; i++) {
            for (int j = 0; j < block; j++) {
                for (int k = 0; k < block; k++) {
                    C_local[i * block + j] +=
                            A_local[i * block + k] * B_local[k * block + j];
                }
            }
        }

        // Shift A left
        MPI_Sendrecv_replace(A_local, block * block, MPI_INT,
                             left, 0, right, 0, comm_2d, MPI_STATUS_IGNORE);

        // Shift B up
        MPI_Sendrecv_replace(B_local, block * block, MPI_INT,
                             up, 0, down, 0, comm_2d, MPI_STATUS_IGNORE);
    }

    // Gather result
    MPI_Gatherv(C_local, block * block, MPI_INT,
                C, sendcounts, displs, block_type,
                0, comm_2d);

    if (rank == 0) {
        end = getticks();
        /*printf("Result Matrix C:\n");
        for (int i = 0; i < N * N; i++) {
            printf("%d ", C[i]);
            if ((i + 1) % N == 0) printf("\n");
        }
         */
        printf("Total time in MPI Process is %lf seconds \n", (double)(end - start) / (double)512000000.0);
    }

    // Cleanup
    MPI_Type_free(&block_type);
    free(A_local);
    free(B_local);
    free(C_local);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}