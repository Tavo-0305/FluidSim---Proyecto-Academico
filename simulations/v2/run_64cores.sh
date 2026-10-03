#!/bin/bash -l
#SBATCH --partition=mpi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name="test"
#SBATCH --output=test.out.%j
#SBATCH --mail-user=gustavo.barbozablanco@ucr.ac.cr
#SBATCH --mail-type=END,FAIL

source ~/bin/conda/etc/profile.d/conda.sh
mamba activate fluidsim 

srun --mpi=pmix python main_mpi.py --N 1024 --t-end 0.05 \
    --sub-directory test_env \
    --period-phys-fields 0.01 \
    --max-elapsed 00:25:00
 

