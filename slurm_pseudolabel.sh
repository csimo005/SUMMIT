#!/bin/bash -l
 
#SBATCH --nodes=1 # Allocate *at least* 5 nodes to this job.
#SBATCH --ntasks=1 # Allocate *at most* 5 tasks for job steps in the job
#SBATCH --cpus-per-task=4 # Each task needs only one CPU
#SBATCH --mem=24G # This particular job won't need much memory
#SBATCH --time=3-00:00:00  # 3 days
#SBATCH --mail-user=csimo005@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="xMUDA Train"
#SBATCH -p vcggpu # You could pick other partitions for other jobs
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=slurm_output/output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

# Place any commands you want to run below
# Activate conda Env
conda activate xmuda2

# Calculate NuScenes Pseudo Labels
python3 xmuda/test.py --cfg=configs/nuscenes/usa_singapore/baseline.yaml --pselab output/73301/nuscenes/usa_singapore/baseline/model_2d_100000.pth output/73301/nuscenes/usa_singapore/baseline/model_3d_100000.pth DATASET_TARGET.TEST "('train_singapore',)" 
python3 xmuda/test.py --cfg=configs/nuscenes/day_night/baseline.yaml --pselab output/76015/nuscenes/day_night/baseline/model_2d_100000.pth output/76015/nuscenes/day_night/baseline/model_3d_100000.pth DATASET_TARGET.TEST "('train_night',)"

# Calculate SemanticKITTI Pseudo Labels
python3 xmuda/test.py --cfg=configs/a2d2_semantic_kitti/baseline.yaml --pselab output/76054/a2d2_semantic_kitti/baseline/model_2d_100000.pth output/76054/a2d2_semantic_kitti/baseline/model_3d_100000.pth  DATASET_TARGET.TEST "('train',)"
