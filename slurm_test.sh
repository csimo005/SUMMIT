#!/bin/bash -l
 
#SBATCH --nodes=1 # Allocate *at least* 5 nodes to this job.
#SBATCH --ntasks=1 # Allocate *at most* 5 tasks for job steps in the job
#SBATCH --cpus-per-task=4 # Each task needs only one CPU
#SBATCH --mem=16G # This particular job won't need much memory
#SBATCH --time=0-12:00:00  # 1 day and 1 minute 
#SBATCH --mail-user=csimo005@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="xMUDA Test"
#SBATCH -p vcggpu # You could pick other partitions for other jobs
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=slurm_output/output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

# Place any commands you want to run below
# Activate conda Env
conda activate xmuda2

# Run USA -> Singapore Tests
python xmuda/test.py --draw --source --cfg=configs/nuscenes/usa_singapore/baseline.yaml output/73301/nuscenes/usa_singapore/baseline/model_2d_100000.pth output/73301/nuscenes/usa_singapore/baseline/model_3d_100000.pth
#python xmuda/test.py --source --cfg=configs/nuscenes/usa_singapore/xmuda.yaml pretrained_weights/nuscenes/usa_singapore/xmuda/model_2d_065000.pth pretrained_weights/nuscenes/usa_singapore/xmuda/model_3d_095000.pth
#python xmuda/test.py --source --cfg=configs/nuscenes/usa_singapore/xmuda_pl.yaml pretrained_weights/nuscenes/usa_singapore/xmuda_pl/model_2d_030000.pth pretrained_weights/nuscenes/usa_singapore/xmuda_pl/model_3d_095000.pth
#
## Run Day -> Night Tests
#python xmuda/test.py --cfg=configs/nuscenes/day_night/baseline.yaml pretrained_weights/nuscenes/day_night/baseline/model_2d_080000.pth pretrained_weights/nuscenes/day_night/baseline/model_3d_090000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda.yaml pretrained_weights/nuscenes/day_night/xmuda/model_2d_075000.pth pretrained_weights/nuscenes/day_night/xmuda/model_3d_090000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl.yaml pretrained_weights/nuscenes/day_night/xmuda_pl/model_2d_055000.pth pretrained_weights/nuscenes/day_night/xmuda_pl/model_3d_070000.pth
#
## Run A2D2 -> SemanticKITTI
#python xmuda/test.py --draw --cfg=configs/a2d2_semantic_kitti/baseline.yaml pretrained_weights/a2d2_semantic_kitti/baseline/model_2d_100000.pth pretrained_weights/a2d2_semantic_kitti/baseline/model_3d_075000.pth
#python xmuda/test.py --draw --cfg=configs/a2d2_semantic_kitti/xmuda.yaml pretrained_weights/a2d2_semantic_kitti/xmuda/model_2d_060000.pth pretrained_weights/a2d2_semantic_kitti/xmuda/model_3d_095000.pth
#python xmuda/test.py --draw --cfg=configs/a2d2_semantic_kitti/xmuda_pl.yaml pretrained_weights/a2d2_semantic_kitti/xmuda_pl/model_2d_080000.pth pretrained_weights/a2d2_semantic_kitti/xmuda_pl/model_3d_085000.pth
