#!/bin/bash

#SBATCH -o /ictstr01/home/aih/laia.mana/project/SLURM_outs/out_%j.job
#SBATCH -e /ictstr01/home/aih/laia.mana/project/SLURM_outs/error_%j.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --nice=10000

source $HOME/.bashrc

#########################
# For quality control
#########################
# conda activate haemorasis
# python3 quality_control.py \
#     --tiles_folder "/ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/hips_sqlite/BM_tiles/BM AI 0551/" \
#     --checkpoint_path "/ictstr01/home/aih/laia.mana/project/quality_control/qc-net" \
#     --output_folder "/ictstr01/home/aih/laia.mana/project/DATA/BM_prova" \
#     --batch_size 4

###############################################################################################################################################

#########################
# For NCA segmentation
#########################
# conda activate msc
# cd /ictstr01/home/aih/laia.mana/project/segmentation_nca/
# python nca_segmentation.py \
#     --input_dir "/ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/hips_sqlite/PB_tiles_preprocessed/PB AI 0360/" \
#     --output_dir "/ictstr01/home/aih/laia.mana/project/PB_wbc/PB AI 0360" 

###############################################################################################################################################

#######################################################
# For feature extraction with other encoders: uni
#######################################################
# conda activate gigapath
# cd /ictstr01/home/aih/laia.mana/project/feature_extraction/
# python feat_extraction_v2.py \
#     --encoder uni2 \
#     --data_path /ictstr01/home/aih/laia.mana/project/DATA/BM_tiles_qc/ \
#     --csv_path /ictstr01/home/aih/laia.mana/project/slideid_label.csv \
#     --out_path /ictstr01/home/aih/laia.mana/project/DATA/BM/feature_extraction/tiles_qc/

############################################
# For feature extraction with dinobloom
############################################
# conda activate dinobloom
# cd /ictstr01/home/aih/laia.mana/project/feature_extraction/
# python feat_extraction_v3.py \
#     --encoder dinobloom-g \
#     --data_path /ictstr01/home/aih/laia.mana/project/DATA/BM_tiles_qc/ \
#     --csv_path /ictstr01/home/aih/laia.mana/project/slideid_label.csv \
#     --out_path /ictstr01/home/aih/laia.mana/project/DATA/BM/feature_extraction/tiles_qc/

###############################################################################################################################################

########################################
# For data preparation (csv generation)
########################################
# conda activate gigapath
# python /ictstr01/home/aih/laia.mana/project/codes/datasets/generate_csv_chipai.py
# python /ictstr01/home/aih/laia.mana/project/codes/datasets/generate_csv_data_sample.py
# python /ictstr01/home/aih/laia.mana/project/codes/datasets/kfold_splits_patient.py

###############################################################################################################################################

#####################################################
# For training aggregation nets with kfold splits
#####################################################
# conda activate gigapath
# cd /ictstr01/home/aih/laia.mana/project/PROJ/feature_aggregation/
# python train_kfold.py \
#             --output /ictstr01/home/aih/laia.mana/project/DATA/BM/feature_aggregation/uni2 \
#             --data /ictstr01/home/aih/laia.mana/project/DATA/BM/kfold/kfold_uni2.csv \
#             --encoder uni2 \
#             --method $method \
#             --kfold $kfold \
#             --lr 0.0001
#     done
# done

################################
# For testing aggregation nets
################################
# conda activate gigapath
# cd /ictstr01/home/aih/laia.mana/project/codes/aggregators_training/
# python test_kfold_AB-MIL.py \
#     --output /ictstr01/home/aih/laia.mana/project/DATA/prova_agr/tiles_uni \
#     --data /ictstr01/home/aih/laia.mana/project/DATA/kfold/kfold_tiles_uni.csv \
#     --checkpoint /ictstr01/home/aih/laia.mana/project/DATA/aggr_train/tiles_uni/AB-MIL/checkpoint_latest_kfold0.pth \
#     --encoder uni2 \
#     --method AB-MIL \
#     --kfold 0

###############################################################################################################################################

sss
