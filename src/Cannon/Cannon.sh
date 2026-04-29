#!/bin/sh

module load xl_r spectrum-mpi cuda

#mpirun -np 4 ./cannon 4000

#./Cannon_CUDA 80
#mpirun -np 4 ./Cannon_MPI 8000
#mpirun -np 16 ./Cannon_MPI 4096

./Cannon_CUDA 4000

#mpirun -np 2 ./cannon 8000
#mpirun -np 4 ./cannon 8000
#mpirun -np 16 ./cannon 8000