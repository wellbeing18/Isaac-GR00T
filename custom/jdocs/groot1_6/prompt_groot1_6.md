

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

some other issues found during inference: 1) robot arms trying to approach objects, but it kinds of either have some grasp timing issue, or vision and arm position mismatch problem(to me feels like it tried to pick up block in the empty space next to it) 2) also if the grasp failed, then the following action will be kind of random 

how to explain those issues, what are the root cause, based on analysis what is the most important steps we need to do next, collect more data, more training steps on the existing data, or after collecting more data, how to analyze data so that we can have sound variance to accelerate model's learning, do we need to finetuning vision model, or do we need to adjust camera. or there is mismatch between model's inference and arm's action which caused the arm's misposition and timing issue, it is kind of overshoot or undershoot, but again you should find more clues from inference data. and help us to identify the most important next step.

investigate for whether both cameras are used in inference properly.

please now combine your suggestions in custom/jdocs/groot1_6/investigation_grasp_issues.md and your suggestions in custom/jdocs/groot1_6/investigation_swing_left_bias.md, what are the ordered priority we need to do? : 

for option A and D you mentioned in custom/jdocs/groot1_6/investigation_swing_left_bias.md: 1) please research and think carefully of using augmented data, it is good for no need to collect more data, but bad if it introduces some unnoticable errors, which could corrupt the model, so make sure you have solid method and verify it carefully with some espisode first before apply to all datasets. also you need to carefully manage those augmented espisodes, so that we have the options to remove or add them back. also after apply it, we should run dataset analsysis script to make sure the dataset is more comprehensive and balanced compared to the baseline dataset or industrial standard. 2) for option D, please explain with more details, how to apply them to dataset or training. you also need to research and compare with other so101 training/finetuning best practises or community feedbacks 

do we need to reference the inference script methods used by getting_started/GR00T_inference.ipynb and getting_started/policy.md, or use client and server, or async method, which can synchronize better model inference and arm execution. or it is not the key issue now based on your research.

also I found an issue during inference start:
Tune action head vlln: True
`use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version

use local run huggineface visualization tool similar to https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fyouliangtan%2Fso101-table-cleanup%2Fepisode_0.

my question above for p3 is for Enable training regularization, after enable it, we should do finetuning from scratch and cannot resume either right?

/home/jrobot/Pictures/local_visualizer.png
/home/jrobot/Pictures/visualizer_whiten.png

/home/jrobot/Pictures/visualizer_closer.png


  First run (15k steps):
  cd /home/jrobot/project/Isaac-GR00T
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate groot
  bash custom/scripts/ver1_6/train_groot_so101_augmented.sh
  Output: outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/

  Resume to 30k (same folder, new checkpoint):
  RESUME_FROM=outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/checkpoint-15000 \
  bash custom/scripts/ver1_6/train_groot_so101_augmented.sh
  Output: outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/checkpoint-30000

  Resume to 45k:
  RESUME_FROM=outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/checkpoint-30000 \
  bash custom/scripts/ver1_6/train_groot_so101_augmented.sh
  Output: outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/checkpoint-45000

python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_augmented_20251219_174824/checkpoint-15000 -n 2

python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
    --checkpoint outputs/groot_1_6_augmented_20251219_174824/checkpoint-15000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 70

RESUME_FROM=outputs/groot_1_6_augmented_20251219_174824/checkpoint-15000 \
  MAX_STEPS=45000 \
  SAVE_STEPS=3000 \
  bash custom/scripts/ver1_6/train_groot_so101_augmented.sh

● Yes, this is an issue. I can see:

  | Phase        | Learning Rate        |
  |--------------|----------------------|
  | End of 15k   | ~1e-12 (nearly zero) |
  | After resume | ~7.5e-05 (jumped!)   |

  What's happening:

  When we resume with MAX_STEPS=45000, the trainer recalculates the LR schedule:
  - Total steps = 45000
  - Current step = 15000 (33% through)
  - Cosine decay at 33% → LR ≈ 7.5e-05

  So it's not restarting at 1e-4, but it's recalculating for the new total, which causes a jump from near-zero to 7.5e-05.

  This explains the loss spike! Going from LR=1e-12 to LR=7.5e-05 is a huge jump.

  Options to fix:

  1. Lower peak LR for resume - Use LEARNING_RATE=1e-5 when resuming
  2. Add small warmup on resume - Use WARMUP_RATIO=0.01 to smooth the transition
  3. Accept it - The model is recovering and training is continuing (it may actually be beneficial since the first run's LR was too low to learn more)

RESUME_FROM=... \
  LEARNING_RATE=1e-5 \
  WARMUP_RATIO=0.01 \
  bash custom/scripts/ver1_6/train_groot_so101_augmented.sh

python custom/scripts/ver1_6/eval_openloop_1_6.py \
      --checkpoint outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000 \
      --num-trajectories 2 \
      --output-dir eval_outputs/checkpoint_45000

      python custom/scripts/ver1_6/eval_openloop_1_6.py \
      --checkpoint outputs/groot_1_6_augmented_20251219_174824/checkpoint-27000 \
      --num-trajectories 2 \
      --output-dir eval_outputs/checkpoint_27000

  RESUME_FROM=outputs/groot_1_6_augmented_20251219_174824/checkpoint-27000 \
  MAX_STEPS=45000 \
  SAVE_STEPS=3000 \
  bash custom/scripts/ver1_6/train_groot_so101_augmented.sh

MAX_STEPS=45000 SAVE_STEPS=1500 bash train_groot_so101_augmented.sh

nohup bash -c 'MAX_STEPS=45000 SAVE_STEPS=1500 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh' > training_45k.log 2>&1 &

nohup bash -c 'MAX_STEPS=45000 SAVE_STEPS=1500 SAVE_TOTAL_LIMIT=15 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh' > /home/jrobot/project/Isaac-GR00T/outputs/training_45k.log 2>&1 &


python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
    --checkpoint outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 70


potential issues:

video format?

  1. Video Backend Compatibility

  | Backend        | AV1 Support | Notes                                        |
  |----------------|-------------|----------------------------------------------|
  | torchcodec     | ✅ Yes      | GR00T default (enforced in experiment.py:41) |
  | torchvision_av | ✅ Yes      | Works, uses PyAV                             |
  | decord         | ❌ No       | Will fail with AV1 codec error               |
  | opencv         | ⚠️ Limited  | May have issues                              |

  2. Training vs Evaluation Backend Mismatch

  From your codebase (custom/jdocs/potential_issues_tutorial_comparison.md:118):
  - Training: uses torchcodec (default)
  - Your custom scripts often use torchvision_av

  This mismatch could potentially cause slightly different frame retrieval for the same timestamp, but both support AV1 correctly.

  3. Frame Retrieval Differences

  Different backends use different logic:
  - torchcodec: Maps timestamps with precision correction (video_utils.py:337-342)
  - torchvision_av: Seeks to keyframe, iterates to target

  Conclusion

  No, the video format is NOT a concern for training performance. Your datasets:
  1. Use AV1 codec - exactly what NVIDIA recommends
  2. Match the SO-101 tuning guide specifications
  3. Are correctly formatted for GR00T's torchcodec backend

  The only potential issue is if you accidentally use decord backend (which doesn't support AV1), but the default torchcodec handles AV1 properly.

python custom/tools/visualize_dataset_v2.py --dataset datasets/so101_pick_place_groot_augmented
python custom/tools/visualize_dataset_v2.py --dataset demo_data/cube_to_bowl_5

python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
  --checkpoint outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000 \
  --task "pick up the blocks and place them on the plate" \
  --duration 70 \
  --record

new investigations:

we did 45k steps training, didn;t see any improvement than 15k training: to me the training steps is not the concern for now, and further verified by groot's example's 1 epoch finetuning

key symptoms:

- misplace or swing or pick up in air issue
  - gerate the traces for inference: [images, text joints] for investigation
  - inference freq or timing mismatch issue 


- timing for gripper
  - assumptions:
    - move too quickly when close to target?
    - open gripper earlier and larger?

