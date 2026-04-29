Due to the nature of fox algorithm requiring a square grid of processors
the number of ranks in an MPI algorithm must be a perfect square. It must
also be a power of 2. This means that for the MPI only implementation the
only viable rank sizes are 1, 4, 16, 64. In the MPI and CUDA implementation, 
in addition to needing a perfect square number of ranks, there also needs 
to be 1 GPU per rank. This makes the only viable rank sizes for MPI/CUDA
1, 4, 16.

This program requires one command line argument which is the exponent of 
the dimension of the matrices to be multiplied. For example, passing in 12
as a command line argument results in the multiplication of two 2^12 X 2^12
matrices. It is recommended that for testing the MPI only implementation
the size of the matrices should be 2^12 X 2^12. For MPI/CUDA it is recommended 
to test on 2^15 X 2^15 matrices.

To compile the MPI only implementation run mpicc -O3 fox_mpi.c -o fox_mpi 
(or whatever you want to call the executable). To compile the MPI/CUDA 
implementation a Makefile is provided. The executable file for this will 
be called fox_mpi_cuda-exe, but can also be changed if desired by editing
the Makefile.

To run these programs it is recommended to create a bash script which loads
in the needed modules - xl_r spectrum-mpi cuda/11.2 - and then using mpirun
to run the executable file. It should look something like the following:
#!/bin/bash
module load xl_r spectrum-mpi cuda/11.2
mpirun -np $SLURM_NPROCS /gpfs/u/home/PCPG/PCPGcnfr/barn/final_project/name_of_executable 12 (or 15 when using GPUs)
