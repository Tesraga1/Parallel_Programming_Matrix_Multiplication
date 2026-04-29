#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include <mpi.h>


typedef unsigned long long ticks;


// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void) {
    unsigned int tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((unsigned long long)tbu0) << 32) | tbl;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("ERROR: Incorrect command line arguments");
        return 0;
    }


    int npes, myrank;
    ticks start_time, end_time;


    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


    // ensure number of processes is a perfect square
    // this is vital for the fox algorithm
    int q = (int)sqrt(npes);
    if (q*q != npes) {
        if (myrank == 0) {
            printf("ERROR: Numbers of processes must be a perfect square\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (myrank == 0) {
        printf("Running MPI program on %d ranks | %d X %d processor grid\n", npes, q, q);
        printf("Matrix Size: 2^%s X 2^%s\n", argv[1], argv[1]);
    }


    int numElements = 1 << atoi(argv[1]);   // n in an n x n matrix
    int block_size = numElements / q;       // m in an m x m submatrix
    int dimension[2] = {q, q};              // size of the processor grid
    int periods[2] = {1, 1};                // indicates that rows and columns wrap around

    // create a communicator and cartesian grid of processes
    // each process is responsible for a chunk of the matrix
    MPI_Comm GRID_COMM;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimension, periods, 1, &GRID_COMM);

    int my_coords[2];
    MPI_Cart_coords(GRID_COMM, myrank, 2, my_coords);
    
    // initialize matrices - matrices are stored in one contiguous chunk of data rather than an array of int pointers
    // local_A[i][j] = local_A[i * block_size + j]
    // rather than splitting up an entire matrix each rank initializes its own chunk - could also be read from a file MPI IO
    int local_num_elements = block_size * block_size;
    int* local_A = malloc(local_num_elements*sizeof(int));
    int* local_B = malloc(local_num_elements*sizeof(int));
    int* local_C = malloc(local_num_elements*sizeof(int));
    int* temp_A = malloc(local_num_elements*sizeof(int));


    // initialize local portion of matrix
    for (int i = 0; i < local_num_elements; i++) {
        local_A[i] = 1;
        local_B[i] = 1;
        local_C[i] = 0;
    }


    // create row and column communicators needed for broadcasting across rows of A and shifting cols of B
    MPI_Comm ROW_COMM;
    MPI_Comm COL_COMM;
    int coords[2];

    coords[0] = 0;
    coords[1] = 1;
    MPI_Cart_sub(GRID_COMM, coords, &ROW_COMM);

    coords[0] = 1;
    coords[1] = 0;
    MPI_Cart_sub(GRID_COMM, coords, &COL_COMM);

    MPI_Barrier(MPI_COMM_WORLD);

    // run Fox algorithm
    start_time = getticks();
    for (int i = 0; i < q; i++) {
        int root = (my_coords[0] + i) % q;

        if (my_coords[1] == root) {
            for(int j = 0; j < local_num_elements; j++) {
                temp_A[j] = local_A[j];
            }
        }

        MPI_Bcast(temp_A, local_num_elements, MPI_INT, root, ROW_COMM);

        // local matrix multiplication
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                int a_val = temp_A[j * block_size + k];
                for (int l = 0; l < block_size; l++) {
                    local_C[j * block_size + l] += a_val * local_B[k * block_size + l];
                }
            }
        }

        // shift local_B upwards in column
        int above, below;
        // Get neighbors for circular shift along dimension 0 (the rows)
        MPI_Cart_shift(COL_COMM, 0, 1, &above, &below);
        MPI_Sendrecv_replace(local_B, local_num_elements, MPI_INT, above, 0, below, 0, COL_COMM, MPI_STATUS_IGNORE);
    }


    // print run time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = getticks();
    if (myrank == 0) {
        printf("Total time running MPI is %lf seconds \n", (double)(end_time - start_time) / (double)512000000.0);
    }


    // check if there were any errors in computations
    for (int i = 0; i < local_num_elements; i++) {
        if (local_C[i] != numElements) {
            printf("PROCESS: %d\n", myrank);
            printf("ERROR: value is %d but should be %d\n\n", local_C[i], numElements);
        }
    }


    // clean up memory and return
    free(local_A);
    free(local_B);
    free(local_C);
    free(temp_A);
    MPI_Finalize();
    return 0;
}