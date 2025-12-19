

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
    --checkpoint outputs/groot_1_6_so101_20251218_201830/checkpoint-30000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 70 \
    --record

resume training
OUTPUT_DIR="/home/jrobot/project/Isaac-GR00T/outputs/groot_1_6_so101_20251218_201830" \
  MAX_STEPS=30000 \
  bash custom/scripts/ver1_6/train_groot_so101_1_6.sh

issues found: 1) robot arms trying to approach objects, but it kinds of either have some grasp timing issue, or vision and arm position mismatch problem(to me feels like it tried to pick up block in the empty space next to it) 2) also if the grasp failed, then the following action will be kind of random 

how to explain those issues, what are the root cause, based on analysis what is the most important steps we need to do next, collect more data, more training steps on the existing data, or after collecting more data, how to analyze data so that we can have sound variance to accelerate model's learning, do we need to finetuning vision model, or do we need to adjust camera. or there is mismatch between model's inference and arm's action which caused the arm's misposition and timing issue, it is kind of overshoot or undershoot, but again you should find more clues from inference data. and help us to identify the most important next step.

investigate for whether both cameras are used in inference properly.

after your investigation abovefor your suggestions in custom/jdocs/groot1_6/investigation_swing_left_bias.md: we can do option B later if required, let's talks about option A and D: 1) please research and think carefully of using augmented data, it is good for no need to collect more data, but bad if it introduces some unnoticable errors, which could corrupt the model, so make sure you have solid method and verify it carefully with some espisode first before apply to all datasets. also you need to carefully manage those augmented espisodes, so that we have the options to remove or add them back. also after apply it, we should run dataset analsysis script to make sure the dataset is more comprehensive and balanced compared to the baseline dataset or industrial standard. 2) for option D, please explain with more details, how to apply them to dataset or training. you also need to research and compare with other so101 training/finetuning best practises or community feedbacks 

Tune action head vlln: True
`use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version