#!/bin/bash
#SBATCH --job-name=summa_all
#SBATCH --output=summa_all_%j.out
#SBATCH --error=summa_all_%j.err
#SBATCH --nodes=1
#SBATCH --partition=el8-rpi
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

module load xl_r spectrum-mpi cuda

OUTPUT="summa_results.csv"
echo "Algorithm,Implementation,MatrixSize,Ranks,Time_sec,GFLOPS" > $OUTPUT

echo "========================================="
echo "SUMMA Algorithm Performance Study"
echo "========================================="
echo ""

# ===== STRONG SCALING: Fixed problem size =====
echo "=== STRONG SCALING: 4096x4096 matrix ==="
MATRIX=4096

# MPI-only
echo "Testing MPI-only (4, 16 ranks)..."
for ranks in 4 16; do
  OUT=$(mpirun -np $ranks ./summa_mpi $MATRIX | grep -E "Time:|GFLOPS")
  TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
  GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
  echo "SUMMA,MPI,$MATRIX,$ranks,$TIME,$GFLOPS" >> $OUTPUT
done

# MPI+CUDA
echo "Testing MPI+CUDA (4, 16 ranks)..."
for ranks in 4 16; do
  OUT=$(mpirun -np $ranks ./summa_mpi_cuda $MATRIX | grep -E "Time:|GFLOPS")
  TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
  GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
  echo "SUMMA,MPI+CUDA,$MATRIX,$ranks,$TIME,$GFLOPS" >> $OUTPUT
done

# CUDA-only (1 GPU, no MPI)
echo "Testing CUDA-only..."
OUT=$(./summa_cuda $MATRIX | grep -E "Time:|GFLOPS")
TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
echo "SUMMA,CUDA,$MATRIX,1,$TIME,$GFLOPS" >> $OUTPUT

echo ""

# ===== WEAK SCALING: Constant work per rank =====
echo "=== WEAK SCALING: 512x512 per rank ==="

# MPI-only
echo "Testing MPI-only weak scaling..."
mpirun -np 4 ./summa_mpi 1024 | tee -a temp.out
OUT=$(cat temp.out | grep -E "Time:|GFLOPS")
TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
echo "SUMMA,MPI,1024,4,$TIME,$GFLOPS" >> $OUTPUT

mpirun -np 16 ./summa_mpi 2048 | tee temp.out
OUT=$(cat temp.out | grep -E "Time:|GFLOPS")
TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
echo "SUMMA,MPI,2048,16,$TIME,$GFLOPS" >> $OUTPUT

# MPI+CUDA
echo "Testing MPI+CUDA weak scaling..."
mpirun -np 4 ./summa_mpi_cuda 1024 | tee temp.out
OUT=$(cat temp.out | grep -E "Time:|GFLOPS")
TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
echo "SUMMA,MPI+CUDA,1024,4,$TIME,$GFLOPS" >> $OUTPUT

mpirun -np 16 ./summa_mpi_cuda 2048 | tee temp.out
OUT=$(cat temp.out | grep -E "Time:|GFLOPS")
TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
echo "SUMMA,MPI+CUDA,2048,16,$TIME,$GFLOPS" >> $OUTPUT

# CUDA-only (different sizes for comparison)
echo "Testing CUDA-only different sizes..."
for size in 1024 2048 4096; do
  OUT=$(./summa_cuda $size | grep -E "Time:|GFLOPS")
  TIME=$(echo "$OUT" | grep "Time:" | awk '{print $2}')
  GFLOPS=$(echo "$OUT" | grep "GFLOPS:" | awk '{print $2}')
  echo "SUMMA,CUDA,$size,1,$TIME,$GFLOPS" >> $OUTPUT
done

rm -f temp.out

echo ""
echo "========================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT"
echo "========================================="

cat $OUTPUT
