# SUMMA Matrix Multiplication - Group Project

## Implementation: 3 Variants

1. **summa_mpi.c** - MPI-only (CPU computation)
2. **summa_mpi_cuda.c/.cu** - MPI+CUDA hybrid (GPU computation)
3. **summa_cuda.cu** - CUDA-only (single GPU baseline)

## Complete Workflow

### Step 1: Compile All Versions

```bash
cd ~/Group_Project

# Load modules
module load xl_r spectrum-mpi cuda

# Compile in scratch (to avoid quota issues)
cd ~/scratch
cp ~/Group_Project/*.c ~/Group_Project/*.cu .

# 1. MPI-only version
mpixlc -O3 summa_mpi.c -o summa_mpi -lm

# 2. CUDA-only version
nvcc -O3 -arch=sm_70 summa_cuda.cu -o summa_cuda

# 3. MPI+CUDA hybrid (use Makefile)
cp ~/Group_Project/Makefile_summa_cuda Makefile
make

# Copy all executables back
cp summa_mpi summa_mpi_cuda summa_cuda ~/Group_Project/
cd ~/Group_Project
```

### Step 2: Run Full Experiments

```bash
chmod +x run_summa_all.sh
sbatch run_summa_all.sh

# Monitor progress
squeue -u PCPGgssl
tail -f summa_all_JOBID.out
```

### Step 3: Get Results

After job completes:
```bash
cat summa_results.csv
```

## What Gets Measured

### Strong Scaling
- Fixed matrix: 4096×4096
- Varying ranks: 4, 16
- All 3 implementations

### Weak Scaling  
- Fixed work per rank: 512×512 block
- Matrix scales with ranks:
  - 4 ranks → 1024×1024
  - 16 ranks → 2048×2048
- All 3 implementations


## Key Constraints

- **Ranks must be perfect squares**: 1, 4, 9, 16, 25, 36, 49, 64...
- **Matrix size must be divisible by √ranks**
- **Example valid configs**:
  - 4 ranks: 1024, 2048, 4096, 8192
  - 16 ranks: 2048, 4096, 8192
  - 64 ranks: 4096, 8192


