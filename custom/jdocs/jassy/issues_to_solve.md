I am currently training groot 1.6 on my soarm101 arm to "pick up the blocks and place them on the plate".
The performance of the model during inference is not great. I think the issue could stem from either the data I collect 
with my soarm101 leader arms or the inference script. Right now, I want to focus on the data and identify any problems
with it. 

Currently, the models performance during inference has 2 key issues.

1. Timing. When the arm approaches a block, a lot of the times, it will bump into the table without opening its gripper.
It seems like it opens the gripper too late and misses the block. I think one of the possible reasons this could be
happening is that during my data collection, I move the arms very quickly and also open the gripper right before I am
going to pick up the cube. Perhaps if I open the gripper very early on and approach the cube with it open I could
partially solve this? And also move a bit slower?

2. Inaccuracy. A lot of the times, the arm will miss its target. When the blocks are on the left and the plate is on the 
right, some of the times the arm will swing very left, missing the blocks entirely, and try to pick up something with its
gripper, but nothing is there and its empty because it overswung. Its performance seems to be slightly better when the 
blocks are on the right and the plate is on the left, but it still misses the block by a couple inches and ends up either
on the left or right of the block. Even when it tries to readjust it by going up and aligning with the block, it still is
not accurate. I think there are a couple possibilities of issues here. Reflecting back to my data collection, I might have
included a bias where 60 percent of the time, I like to start with the left-most block and pick that up first. So if the
blocks are spread out on the left, perhaps its learned to just swing to the very far left and start there? Another thing
is, I was trying to include some failed attempts(not purposefully, but when it happened, I kept them in) where if I
missed the cube, I would readjust and try again. Every episode, I pick up 6 blocks. I usually make a mistake with 1 block
if any, and this happens for around 30 out of the 70 episodes I recorded. It could be that I don't make enough mistakes
or made too many that is causing it to learn to miss the block?

There are also a couple other things I wonder if they are affecting the performance that I notice when comparing with
groot's demo videos online.
1. should I be moving the arm at a fast pace but slow down when I am picking up the object?
2. there head cameras seem to be lower than mine. Is that causing the missing the target issue?

Here is what I want you to do:

1. analyze my current problems and come up with your own hypothesis's for why it could be happening based on research
from succesful attempts for finetuning groot or other models like pi0.5 on robot embodiments and try to explain why this
is happening. You do not have to listen to or confirm my speculations.
2. take my collected dataset: /home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot_augmented and a "baseline"
dataset taken from online like : https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fyouliangtan%2Fso100_strawberry_grape%2Fepisode_25 or demo_data/cube_to_bowl_5 an use statistics and graphs to analyze the differences in things like joints, trajectories, speed, etc(those were examples, you do not have to go out of your way to strictly follow them) and identify any differences. you can use thigns like dataset visualizer or draw png images
3. do the same thing as step 2 but see if my hypothesis's were accurate
5. provide next steps for collecting data and things to looks out for/do



-open gripper earlier and wider
-stop before closing gripper
-slow down approach

-get rid of starting from the left or rectangle bias
-get rid of starting from the farthest cube bias

-try to move wrist pan more
-move arm back more
-
-check if I can zoom in the camera
-don't move the plate