
mpi-cuda: fox_mpi_cuda.c fox_mpi_cuda.cu
	mpixlc -O3 fox_mpi_cuda.c -c -o fox_mpi_cuda-xlc.o
	nvcc -O3 -arch=sm_70 fox_mpi_cuda.cu -c -o fox_mpi_cuda-nvcc.o 
	mpixlc -O3 fox_mpi_cuda-xlc.o fox_mpi_cuda-nvcc.o -o fox_mpi_cuda-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ 
