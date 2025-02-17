#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
now=$(date +"%Y%m%d_%H%M%S")

epoch=300
bs=4
gpus=4
lr=0.000005
encoder=vitl
dataset=soccernet
img_size=294
min_depth=0.001
max_depth=250 # 80 for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=/data/ipad_3d/dmy/depthanything_output/logs/6_MSELoss_255_300 # exp/vkitti

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20593 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
