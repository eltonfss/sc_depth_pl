# absolute path that contains all datasets
DATA_ROOT=/home/jussarafigueiredodesouzasoares/sc_depth_pl/data

# ddad
#DATASET=ddad
#RESULTS_DIR=results/$DATASET/model_v3/depth
#GT_DIR=$DATA_ROOT/$DATASET/testing/depth
#SEG_MASK=$DATA_ROOT/$DATASET/testing/seg_mask

# kitti
DATASET=kitti
RESULTS_DIR=results/$DATASET/debugging/model_v3/depth
GT_DIR=$DATA_ROOT/$DATASET/testing/depth
SEG_MASK=$DATA_ROOT/$DATASET/testing/seg_mask


# run evaluation
export CUDA_VISIBLE_DEVICES=0
python eval_depth.py \
--dataset $DATASET \
--pred_depth=$RESULTS_DIR \
--gt_depth=$GT_DIR \
--seg_mask=$SEG_MASK
