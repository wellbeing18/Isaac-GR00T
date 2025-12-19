

scripts to create:

1) data conversion

2) zero shot verification
- will download groot 1.6 
- verify env


3) so101 finetuning
- no lora
- 

1) verification


5) inference
- accelerate: https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md#component-wise-breakdown
- 


MAX_STEPS=22000 SAVE_STEPS=2000 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh

python custom/scripts/ver1_6/eval_openloop_1_6.py \
      --checkpoint outputs/groot_1_6_so101_20251218_201830/checkpoint-18000 \
      --num-trajectories 2 \
      --output-dir eval_outputs/checkpoint_18000

update eval openloop script to 

python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
    --checkpoint outputs/groot_1_6_so101_20251218_201830/checkpoint-18000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 30 \
    --record

resume training
OUTPUT_DIR="/home/jrobot/project/Isaac-GR00T/outputs/groot_1_6_so101_20251218_201830" \
  MAX_STEPS=30000 \
  bash custom/scripts/ver1_6/train_groot_so101_1_6.sh