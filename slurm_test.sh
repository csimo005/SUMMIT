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
mamba activate xmuda2

# Run USA -> Singapore Tests
#printf "****************************************\n"
#printf "*                                      *\n"
#printf "*           USA -> Singapore           *\n"
#printf "*                                      *\n"
#printf "****************************************\n"

#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/125633/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/125633/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/81878/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/81878/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/81879/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/81879/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/81880/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/81880/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth

## baseline methods
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/baseline.yaml output/73301/nuscenes/usa_singapore/baseline/model_2d_100000.pth output/73301/nuscenes/usa_singapore/baseline/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl.yaml output/75988/nuscenes/usa_singapore/xmuda_pl/model_2d_100000.pth output/75988/nuscenes/usa_singapore/xmuda_pl/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/SHOT_1.yaml output/82780/nuscenes/usa_singapore/SHOT_1/model_2d_100000.pth output/82780/nuscenes/usa_singapore/SHOT_1/model_3d_100000.pth
#
## our methods
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/81599/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/81599/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/83513/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/83513/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/83474/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/83474/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/83519/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/83519/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/83508/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/83508/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/83509/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/83509/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl_SF.yaml output/76467/nuscenes/usa_singapore/xmuda_pl_SF/model_2d_100000.pth output/76467/nuscenes/usa_singapore/xmuda_pl_SF/model_3d_100000.pth

## Run Day -> Night Tests
#printf "****************************************\n"
#printf "*                                      *\n"
#printf "*             Day -> Night             *\n"
#printf "*                                      *\n"
#printf "****************************************\n"
#
## baseline methods
#python xmuda/test.py --cfg=configs/nuscenes/day_night/baseline.yaml output/76015/nuscenes/day_night/baseline/model_2d_100000.pth output/76015/nuscenes/day_night/baseline/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl_SF.yaml output/76052/nuscenes/day_night/xmuda_pl_SF/model_2d_100000.pth output/76052/nuscenes/day_night/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/SHOT.yaml output/83014/nuscenes/day_night/SHOT/model_2d_100000.pth output/83014/nuscenes/day_night/SHOT/model_3d_100000.pth
#
## our methods
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl_SF.yaml output/81600/nuscenes/day_night/xmuda_pl_SF/model_2d_100000.pth output/81600/nuscenes/day_night/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl_SF.yaml output/83535/nuscenes/day_night/xmuda_pl_SF/model_2d_100000.pth output/83535/nuscenes/day_night/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl_SF.yaml output/83488/nuscenes/day_night/xmuda_pl_SF/model_2d_100000.pth output/83488/nuscenes/day_night/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl_SF.yaml output/83562/nuscenes/day_night/xmuda_pl_SF/model_2d_100000.pth output/83562/nuscenes/day_night/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda_pl_SF.yaml output/81717/nuscenes/day_night/xmuda_pl_SF/model_2d_100000.pth output/81717/nuscenes/day_night/xmuda_pl_SF/model_3d_100000.pth

## Run A2D2 -> SemanticKITTI
#printf "****************************************\n"
#printf "*                                      *\n"
#printf "*        A2D2 -> SemanticKITTI         *\n"
#printf "*                                      *\n"
#printf "****************************************\n"
#
## baseline methods
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/baseline.yaml output/76054/a2d2_semantic_kitti/baseline/model_2d_100000.pth output/76054/a2d2_semantic_kitti/baseline/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/76067/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/76067/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/83449/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/83449/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#
## our methods
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/81601/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/81601/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/139432/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/139432/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/139438/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/139438/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/83578/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/83578/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/83496/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/83496/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/83552/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/83552/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl_SF.yaml output/76569/a2d2_semantic_kitti/xmuda_pl_SF/model_2d_100000.pth output/76569/a2d2_semantic_kitti/xmuda_pl_SF/model_3d_100000.pth

## 2D: Day, 3D: Night -> SemanticKITTI
printf "****************************************\n"
printf "*                                      *\n"
printf "*  2D: Day, 3D: Night -> SemanticKITTI *\n"
printf "*                                      *\n"
printf "****************************************\n"
python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/109735/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/109735/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/139556/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/139556/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/139557/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/139557/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth


#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/baseline_day.yaml output/109325/nuscenes_lidarseg_semantic_kitti/day_night/baseline_day/model_2d_100000.pth output/109329/nuscenes_lidarseg_semantic_kitti/day_night/baseline_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/114735/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/114735/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/shot_day.yaml output/114786/nuscenes_lidarseg_semantic_kitti/day_night/shot_day/model_2d_100000.pth output/114786/nuscenes_lidarseg_semantic_kitti/day_night/shot_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/109481/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/109481/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/109735/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/109735/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/109751/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/109751/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/109743/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/109743/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day.yaml output/109755/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_2d_100000.pth output/109755/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_day/model_3d_100000.pth

## 2D: Night, 3D: Day -> SemanticKITTI
#printf "****************************************\n"
#printf "*                                      *\n"
#printf "*  2D: Night, 3D: Day -> SemanticKITTI *\n"
#printf "*                                      *\n"
#printf "****************************************\n"
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/baseline_night.yaml output/109329/nuscenes_lidarseg_semantic_kitti/day_night/baseline_night/model_2d_100000.pth output/109325/nuscenes_lidarseg_semantic_kitti/day_night/baseline_day/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night.yaml output/114736/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_2d_100000.pth output/114736/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/shot_night.yaml output/114787/nuscenes_lidarseg_semantic_kitti/day_night/shot_night/model_2d_100000.pth output/114787/nuscenes_lidarseg_semantic_kitti/day_night/shot_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night.yaml output/109482/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_2d_100000.pth output/109482/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night.yaml output/109736/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_2d_100000.pth output/109736/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night.yaml output/109752/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_2d_100000.pth output/109752/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night.yaml output/109744/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_2d_100000.pth output/109744/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night.yaml output/109756/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_2d_100000.pth output/109756/nuscenes_lidarseg_semantic_kitti/day_night/xmuda_night/model_3d_100000.pth

## 2D: USA, 3D: Singapore -> SemanticKITTI
#printf "*****************************************\n"
#printf "*                                       *\n"
#printf "*2D: USA, 3D: Singapore -> SemanticKITTI*\n"
#printf "*                                       *\n"
#printf "*****************************************\n"
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/baseline_usa.yaml output/109330/nuscenes_lidarseg_semantic_kitti/usa_singapore/baseline_usa/model_2d_100000.pth output/109331/nuscenes_lidarseg_semantic_kitti/usa_singapore/baseline_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml output/114733/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_2d_100000.pth output/114733/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/shot_usa.yaml output/114784/nuscenes_lidarseg_semantic_kitti/usa_singapore/shot_usa/model_2d_100000.pth output/114784/nuscenes_lidarseg_semantic_kitti/usa_singapore/shot_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml output/109483/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_2d_100000.pth output/109483/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml output/109737/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_2d_100000.pth output/109737/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml output/109753/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_2d_100000.pth output/109753/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml output/109747/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_2d_100000.pth output/109747/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa.yaml output/109757/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_2d_100000.pth output/109757/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_usa/model_3d_100000.pth

## 2D: Singapore, 3D: USA -> SemanticKITTI
#printf "*****************************************\n"
#printf "*                                       *\n"
#printf "*2D: Singapore, 3D: USA -> SemanticKITTI*\n"
#printf "*                                       *\n"
#printf "*****************************************\n"
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/baseline_singapore.yaml output/109331/nuscenes_lidarseg_semantic_kitti/usa_singapore/baseline_singapore/model_2d_100000.pth output/109330/nuscenes_lidarseg_semantic_kitti/usa_singapore/baseline_usa/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore.yaml output/114734/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_2d_100000.pth output/114734/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/shot_singapore.yaml output/114785/nuscenes_lidarseg_semantic_kitti/usa_singapore/shot_singapore/model_2d_100000.pth output/114785/nuscenes_lidarseg_semantic_kitti/usa_singapore/shot_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore.yaml output/109484/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_2d_100000.pth output/109484/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore.yaml output/109738/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_2d_100000.pth output/109738/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore.yaml output/109754/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_2d_100000.pth output/109754/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore.yaml output/109748/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_2d_100000.pth output/109748/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_3d_100000.pth
#python xmuda/test.py --cfg=configs/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore.yaml output/109758/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_2d_100000.pth output/109758/nuscenes_lidarseg_semantic_kitti/usa_singapore/xmuda_singapore/model_3d_100000.pth
