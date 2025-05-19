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

############################
# For data preparation (csv generation)
############################
# conda activate gigapath
# python /ictstr01/home/aih/laia.mana/project/codes/datasets/generate_csv_chipai.py
# python /ictstr01/home/aih/laia.mana/project/codes/datasets/generate_csv_data_sample.py
# python /ictstr01/home/aih/laia.mana/project/codes/datasets/kfold_splits_patient.py

###############################################################################################################################################

########################################################################
# For feature extraction with other encoders: uni, virchow, gigapath
########################################################################
# conda activate gigapath
# cd /ictstr01/home/aih/laia.mana/project/codes/feat_extraction/
# python feat_extraction_v2.py \
#     --encoder uni \
#     --data_path /ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/hips_sqlite/PB_tiles_preprocessed/ \
#     --csv_path /ictstr01/home/aih/laia.mana/project/slideid_label.csv \
#     --out_path /ictstr01/home/aih/laia.mana/project/DATA/feature_extraction/features_tiles_preprocessed/ 

############################################
# For feature extraction with dinobloom
############################################
# conda activate dinobloom
# cd /ictstr01/home/aih/laia.mana/project/codes/feat_extraction/
# python feat_extraction_v3.py \
#     --encoder dinobloom-g \
#     --data_path /ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/hips_sqlite/PB_tiles_preprocessed/ \
#     --csv_path /ictstr01/home/aih/laia.mana/project/slideid_label.csv \
#     --out_path /ictstr01/home/aih/laia.mana/project/DATA/feature_extraction/features_tiles_preprocessed/ 

###############################################################################################################################################

#####################################################
# For training aggregation nets with kfold splits
#####################################################
# conda activate gigapath
# cd /ictstr01/home/aih/laia.mana/project/codes/aggregators_training/
# python train_kfold.py \
#     --output /ictstr01/home/aih/laia.mana/project/DATA/prova_agr_train/tiles_uni \
#     --data /ictstr01/home/aih/laia.mana/project/DATA/kfold/kfold_tiles_uni.csv \
#     --encoder uni2 \
#     --method transMIL \
#     --kfold 0 \
#     --lr 0.0001

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


conda activate gigapath
cd /ictstr01/home/aih/laia.mana/project/attention_visualization/

ENCODER1=uni2
ENCODER2=dinobloom-g

METHOD1=transMIL
METHOD2=AB-MIL

for kfold in {0..4}; do
    for method in $METHOD1 $METHOD2; do
        python run_feature_aggregation.py \
            --feature_aggregation_path /ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/f_aggregation/$ENCODER1 \
            --kfold_data /ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/kfold/kfold_uni.csv \
            --encoder $ENCODER1 \
            --method $method \
            --kfold $kfold \
            --lr 0.0001
    done
done


# for kfold in {0..4}; do
#     for method in $METHOD1 $METHOD2; do
#         echo "Running inference for fold $kfold, method $method, file $FILE, encoder $enc"

#         python train_kfold.py \
#             --output /ictstr01/home/aih/laia.mana/project/DATA/aggr_train/$DIR \
#             --data /ictstr01/home/aih/laia.mana/project/DATA/kfold/$FILE \
#             --encoder $enc \
#             --method $method \
#             --kfold $kfold \
#             --lr 0.0001
#     done
# done