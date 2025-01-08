#!/bin/bash
#SBATCH --partition=parallel
#SBATCH --nodes=4                                     # node count
#SBATCH --ntasks=4                                    # total number of tasks for oe node
#SBATCH --cpus-per-task=12                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1900MB                          # memory per cpu-core (2G is default)
#SBATCH --time=01:00:00                               # total run time limit (HH:MM:SS)

source ~/software/init-conda
conda activate bera

set -x

NNODES=$SLURM_NNODES
NWORKERS=$((12*${NNODES}-2)) # One core for the scheduler and another one for the python script
MEM_PER_WORKER=1900MB # parallel node has up to 24GB Memory

SCHEDULER_FILE="dask_scheduler.json"

# Start the scheduler
srun --ntasks=1 --nodes=1 --exclusive dask scheduler --scheduler-file $SCHEDULER_FILE &

# Start the workers
srun --ntasks=$NWORKERS --nodes=${NNODES} --exclusive  dask worker --scheduler-file $SCHEDULER_FILE --memory-limit $MEM_PER_WORKER &

sleep $((240*60)) # 240 minutes, adjust to the walltime
