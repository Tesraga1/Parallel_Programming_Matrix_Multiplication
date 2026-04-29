How to Run Cannon

N -> Number of elements,
ranks -> Number of MPI ranks

Cannon_MPI.c:
mpirun -np [ranks] ./Cannon_MPI [N]
(N*N must be a perfect square)

Cannon_CUDA.cu:
./Cannon_CUDA N

MPI_cannon.c with Cuda_Cannnon.cu:
Combine
mpicc -c MPI_cannon.c , nvcc -c Cuda_Cannon.cu, 
mpixlc -O3 Cuda_Cannon.o MPI_cannon.o -o cannon -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
Run: mpirun -np [rank] ./cannon [N]