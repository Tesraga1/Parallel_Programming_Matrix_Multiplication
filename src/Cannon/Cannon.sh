#!/bin/sh

module load xl_r spectrum-mpi cuda

mpirun -np 4 ./cannon
#./Cannon_CUDA 80
#mpirun -np 4 ./Cannon_MPI