# absolute path that contains all datasets
DATA_ROOT=/home/jussarafigueiredodesouzasoares/sc_depth_pl/data

# # kitti
INPUT=$DATA_ROOT/kitti/testing/color
OUTPUT=results/kitti/debugging
CONFIG=configs/v3/kitti_raw.txt
CKPT=ckpts/kitti_scv3/version_13/last.ckpt

# # nyu
# INPUT=$DATA_ROOT/nyu/testing/color
# OUTPUT=results/nyu
# CONFIG=configs/v2/nyu.txt
# CKPT=ckpts/nyu_scv2/epoch=101-val_loss=0.1580.ckpt

#INPUT=$DATA_ROOT/ddad/testing/color
#OUTPUT=results/ddad
#CONFIG=configs/v3/ddad.txt
#CKPT=ckpts/ddad_scv3/epoch=99-val_loss=0.1438.ckpt

# run
export CUDA_VISIBLE_DEVICES=0
python inference.py --config $CONFIG \
--input_dir $INPUT --output_dir $OUTPUT \
--ckpt_path $CKPT --save-vis --save-depth
