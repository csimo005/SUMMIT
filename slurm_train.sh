#!/bin/bash -l
 
#SBATCH --nodes=1 # Allocate *at least* 5 nodes to this job.
#SBATCH --ntasks=1 # Allocate *at most* 5 tasks for job steps in the job
#SBATCH --cpus-per-task=4 # Each task needs only one CPU
#SBATCH --mem=24G # This particular job won't need much memory
#SBATCH --time=3-00:00:00  # 3 days
#SBATCH --mail-user=csimo005@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="nuscenes"
#SBATCH -p vcggpu # You could pick other partitions for other jobs
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=slurm_output/output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

# Place any commands you want to run below
# Activate conda Env
echo $CUDA_VISIBLE_DEVICES
echo $SLURMD_NODENAME
mamba activate xmuda2
python3 xmuda/train_xmuda_SF.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml OUTPUT_DIR output/$SLURM_JOBID/@
