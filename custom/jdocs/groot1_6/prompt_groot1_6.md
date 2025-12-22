

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

we did 45k steps training(open loop eval: eval_outputs/checkpoint_45000), didn't see any improvement than 15k training in inference performance, though the open loop eval is better than 15k's(eval_outputs/checkpoint_15000): to me the training steps is not the concern for now, and further verified by groot's example's 1 epoch finetuning, and its open loop eval performance(similar to ours and somehow worse), but its close loop/robot arm inference performs much better. we need to investigate towards the root cause of this. let's focus on the first key symptom:

- misplace or swing or pick up in air issue
  - i suspect the issue is in inference process
  - my suggestion: I suggest you write a new inference script based on custom/scripts/ver1_6/infer_groot_so101_1_6.py, which can generate the traces during inference: sequence of input at each step: [images, text, joints states], the prediction output [actions seq], together with the time of inference generate, the arm execution, and later new inference start up time, etc.  so that we can inspect them for investigation to help us understand the root cause of the symptom, is it due to inference freq or timing mismatch issue, or camera view with arm action timing mismatch issue, or other issues we haven't realized yet

# 1. Collect traces
python custom/scripts/ver1_6/infer_groot_so101_trace.py \
    --checkpoint outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 30

# 2. Analyze traces
python custom/scripts/ver1_6/analyze_inference_trace.py \
    --trace-dir outputs/inference_traces/trace_20251221_214617 \
    --show-plots


key symptoms:

- move gripper to the wrong place where there is no block or pick up in air issue
  - gerate the traces during inference: sequence of input at each step: [images, text, joints states], the prediction output [actions seq], together with the time of inference generate, the arm execution, and later new inference start up time, etc.  so that we can inspect them for investigation to help us understand the root cause of the symptom, is it due to inference freq or timing mismatch issue, or camera view with arm action timing mismatch issue, or other issues we haven't realized yet


- timing for gripper
  - assumptions:
    - move too quickly when close to target?
    - open gripper earlier and larger?

- lighting
  - hue & bright light
- stable plate

abolute actions
- retrain or use 1.5?

stats.json:
- groot 1.6
  - using pretrained stats or post-trained?

dagger?

test time and train time rtc 

For GR00T N1.6, we conduct more complex real-world robot experiments than GR00T N1.5, requiring long-horizon reasoning, dexterity, and multi-tasking abilities. When scaling up real-world experiments, we incorporate various lessons learned from the robot learning community to improve model success rates during rollouts.

Relative actions are used as the default action space for most embodiments. Our experiments show that relative actions produce smoother and more accurate motions than absolute actions. However, with small datasets, relative actions are prone to error accumulation, which impacts correction ability.
Pretrained statistics can improve performance when the task distribution is similar to the pretraining data; otherwise, the model may underfit, so we use post-training statistics when distributions differ.
GR00T N1.6 converges faster than GR00T N1.5, leading to smoother actions, but requires more careful tuning to prevent overfitting. We apply stronger state regularization, additional data augmentations, and co-training with pretraining data to regularize the model during post-training.
Iterative DAgger effectively improves model performance; this is recommended to be used when the model is underperforming in real-world experiments.
Test-time and train-time RTC provide performance boosts to motion smoothness and robustness during asynchronous rollouts. We employ this technique in Unitree G1 and Bimanual YAM experiments.
Multi-task language following and out-of-distribution task generalization continue to be challenging for current VLA models. More fine-grained subtask annotation can improve language following, but not yet reaching robust generalization. This will be a continuous effort in future research.

# training steps
Reference: HF Blog Dataset (so100-table-cleanup)

  | Parameter          | Value  |
  |--------------------|--------|
  | Samples/Frames     | 47,513 |
  | Episodes           | 80     |
  | FPS                | 30     |
  | Default Batch Size | 32     |
  | Recommended Steps  | 10,000 |

  HF Blog Compute:
  - 10K steps × 32 batch = 320,000 samples seen
  - Epochs: 320K / 47.5K = ~6.7 epochs

  ---
  Your Dataset (so101_pick_place_groot)

  | Parameter       | Value  |
  |-----------------|--------|
  | Samples/Frames  | 60,271 |
  | Episodes        | 70     |
  | Your Batch Size | 8      |

  ---
  Recalculated Steps for Your Setup

  Method 1: Match Total Samples Seen (320K)

  320,000 samples / 8 batch = 40,000 steps

  Method 2: Match Epochs (~6.7 epochs)

  Steps per epoch = 60,271 / 8 = 7,534 steps/epoch
  6.7 epochs × 7,534 = ~50,500 steps

  ---
  Revised Recommendation

  | Target                | Steps (batch 8) |
  |-----------------------|-----------------|
  | Match HF Blog compute | 40,000 steps    |
  | Match HF Blog epochs  | 50,000 steps    |


before I run new script, questions: 1) your investigation direction and goal should be to explain the symptoms I mentioned above, your analysis above to me is kind of off the track, we need to dig from those traces to find and infer the potential 
root causes: is it due to model's prediction quality, or it is due the synchronization problem between model and arm, or it is some bugs in script, etc 2)  did you check the images input, you should carefully go through the images(what model sees at 
inference times), then analyze its predicted actions whether they help move the arms approach it or not, then compare the next input's image and the inference output, etc, you need to follow the trace to understand what is going now, only through this
 you can spot and identify the potential issue 3) you kept saying arm executes 16 actions before the next inference, are you sure? to my understanding, groot model generates 16 actions for each inference, but only send 8 of them for arms to execute? 
also this is also adjustable through parameter setting change? 4) for the timeline staleness issue you are talking, first I don't get what you are talking, if you think you are still right after my below argument you can explain it better to me. my 
argument is: the training is groot model based on input(images, state, txt) to generate 16 actions(traj), at inference time, first groot by default execute 8 actions instead of 16(you need to verify this), second let's still with your 16 actions 
execution case above, what is your point, you mean arm should only execute 1 action instead of 16? third, I think the key question is after model did first inference, was it starting to do the next inference right away(at 130ms for example) or it 
waits arms's execution finished(at 660ms for example), then to do its second inference, 5) you need to do more research after those investigation and analysis to what inference groot or lerobot(for groot): 
scripts/deployment/standalone_inference_script.py, scripts/deployment/GR00T_inference_timing.ipynb, scripts/deployment/*, getting_started/GR00T_inference.ipynb, or lerobot(groot)'s inference method: lerobot-record \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM1 \
  --robot.right_arm_port=/dev/ttyACM0 \
  --robot.id=bimanual_follower \
  --robot.cameras='{ right: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    left: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \
  --display_data=true \
  --dataset.repo_id=<user>/eval_groot-bimanual  \
  --dataset.num_episodes=10 \
  --dataset.single_task="Grab and handover the red cube to the other arm"
  --policy.path=<user>/groot-bimanual # your trained model
  --dataset.episode_time_s=30
  --dataset.reset_time_s=10, as in https://huggingface.co/docs/lerobot/groot 
  

Fix 1: Add Recovery Demonstrations to Training Data

  What you need to record:

  Record new demonstrations where you deliberately start with the arm away from the blocks, then move it toward them:

  New Episode Type:
  1. Move arm far to the left (ShPan = -60° or more)
  2. Wrist camera sees empty table
  3. Head camera shows blocks clearly
  4. YOU demonstrate: move arm back toward blocks (ShPan → 0°)
  5. Continue with normal pick and place

  Concrete steps:

  1. Before recording, manually move the arm to various "wrong" positions:
    - Far left (ShPan = -60°)
    - Far right (ShPan = +60°)
    - Too high, too low
  2. Record yourself moving it back to the blocks and completing the task
  3. Mix these "recovery" episodes into your training dataset (maybe 20-30% of episodes)

  Example script modification for recording:
  # In your recording script, add starting positions:
  RECOVERY_START_POSITIONS = [
      [-60, 50, -50, 80, -90, 30],  # Far left
      [60, 50, -50, 80, -90, 30],   # Far right  
      [0, 20, -30, 60, -60, 30],    # Too high
  ]

  # Randomly start some episodes from these positions
  if random.random() < 0.3:  # 30% of episodes
      start_pos = random.choice(RECOVERY_START_POSITIONS)
      robot.move_to(start_pos)
      print("Starting from recovery position - demonstrate moving back to blocks")

  ---
  Fix 2: Data Augmentation (No New Recording Needed)

  Instead of recording new data, augment existing trajectories by starting from middle states:

  The idea:
  Original trajectory:
    State 0 → State 1 → State 2 → ... → State 100 (done)

  Augmented trajectories (created automatically):
    State 30 → State 31 → ... → State 100 (start from middle)
    State 50 → State 51 → ... → State 100 (start from different point)
    State 70 → State 71 → ... → State 100 (start near end)

  This teaches the model: "If you find yourself in State 50, here's what to do next."

  Concrete implementation:

  # Add to your data loading/preprocessing

  def augment_trajectory_with_random_starts(trajectory, num_augmented=3):
      """
      Create additional training samples by starting from random points in trajectory.
      
      This teaches the model recovery behavior - what to do if it's already
      partway through a task.
      """
      augmented = [trajectory]  # Keep original

      traj_len = len(trajectory)
      for _ in range(num_augmented):
          # Pick random start point (not too close to end)
          start_idx = random.randint(0, int(traj_len * 0.7))

          # Create sub-trajectory from this point
          sub_traj = trajectory[start_idx:]

          if len(sub_traj) >= 10:  # Only keep if long enough
              augmented.append(sub_traj)

      return augmented

  ---
  Why These Fixes Work

  The problem is distribution mismatch:

  | During Training                     | During Inference              |
  |-------------------------------------|-------------------------------|
  | Arm always starts near blocks       | Arm might drift far away      |
  | Wrist always sees something         | Wrist might see empty table   |
  | Head camera view = similar to wrist | Head camera view ≠ wrist view |

  Both fixes expand the training distribution to include "recovery" states, so the model learns:
  - "When wrist sees nothing but head sees blocks on the right → move right"
  - "When I'm far from blocks → navigate back using head camera"

  ---
  Which Fix Should You Try First?

  1. Fix 2 (Data Augmentation) - Try this first
    - No new recording needed
    - Just modify your data loading
    - Can test quickly
  2. Fix 1 (Recovery Demonstrations) - If Fix 2 isn't enough
    - Record 10-20 new episodes starting from "lost" positions
    - More effort but more direct training signal

  Would you like me to write the actual data augmentation code for your training pipeline?


now we are encountering bad inference performance for our latest finetuned groot 1.6 model: outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000, using dataset datasets/so101_pick_place_groot_augmented. I did open loop eval and it looks good to me: eval_outputs/checkpoint_45000, but when we do inference, the performance is bad: cannot finish to pick up any blocks(though it looks trying to approach the blocks and did some trial and error), but it shows some bad behaviors to me: 1) gripper timing is not right(which I think the reason could in dataset which we think we know how to fix) 2) arm aimed for a block but ended with a wrong place next to the block which failed the task 3) the arm sometimes move away from blocks and never recover, or the arm is trying to grip blocks in the air or places there is no blocks.

in order to investigate and diagnose to find the root cause, we created scripts: custom/scripts/ver1_6/infer_groot_so101_trace.py and custom/scripts/ver1_6/analyze_inference_trace.py, which help generate input/outputs(prediction) traces for investigation. and claude has generated its investigation in custom/jdocs/groot1_6/5_inference_mismatch_investigation.md, which doesn't quite convince me, I think there could be some other potential issues.

some inference traces are:
- swing to the far left where there is no block, and never return: outputs/inference_traces/trace_20251221_214617
- grip in wrong places or in the air: outputs/inference_traces/trace_20251221_214755 and outputs/inference_traces/trace_20251221_215113.

currently I think the possible issues are in training dataset(horizontally flipped augmented dataset: datasets/so101_pick_place_groot_augmented), or in our training script custom/scripts/ver1_6/train_groot_so101_augmented.sh(or its modality, stats, or cfg?), or in inference: script custom/scripts/ver1_6/infer_groot_so101_1_6.py or custom/scripts/ver1_6/infer_groot_so101_trace.py. so you could to search and research to help us find the root cause.

for inference, you can also reference groot or lerobot(for groot) inference scripts below: 
scripts/deployment/standalone_inference_script.py, scripts/deployment/GR00T_inference_timing.ipynb, scripts/deployment/*, getting_started/GR00T_inference.ipynb, or lerobot(groot)'s inference method: lerobot-record \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM1 \
  --robot.right_arm_port=/dev/ttyACM0 \
  --robot.id=bimanual_follower \
  --robot.cameras='{ right: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    left: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \
  --display_data=true \
  --dataset.repo_id=<user>/eval_groot-bimanual  \
  --dataset.num_episodes=10 \
  --dataset.single_task="Grab and handover the red cube to the other arm"
  --policy.path=<user>/groot-bimanual # your trained model
  --dataset.episode_time_s=30
  --dataset.reset_time_s=10, as in https://huggingface.co/docs/lerobot/groot 

you don't make any changes, but write your investigation report to: custom/jdocs/groot1_6/7_gemini__inference_mismatch_investigation.md