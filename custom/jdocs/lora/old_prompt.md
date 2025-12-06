The linter errors (numpy import, float.tolist) are minor type checking issues that won't affect runtime in this environment where numpy is installed (as seen in other files).
I have now:
Fixed convert_lerobot_v3_to_groot.py:
Updated generate_episodes_jsonl to include task_index in the episode metadata, reading it from the parquet source if available, or defaulting to 0. This fixes the "Missing task_index" issue reported by Claude.
Fixed combine_groot_datasets.py:
Rewrote the combination logic to preserve the "Chunked" structure of LeRobot datasets.
Instead of flattening everything to episode_XXX, it now calculates chunk_offset based on chunk_size (preserving alignment).
It moves parquet files to data/chunk-{new_chunk}/episode_{new_ep}.parquet.
It moves video files to videos/{key}/chunk-{new_chunk}/....
This avoids the need to re-encode/split video files, as we simply shift the chunks to new indices.
Updated info.json to use the chunked data_path and video_path patterns.


python custom/scripts/infer_groot_so101.py \
      --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251204_215410259/best \
      --task "pick the red cube from the table"

python custom/scripts/infer_groot_so101.py \
      --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251204_215410259/best \
      --task "pick the red cube from the table" \
      --action-horizon 6 \
      --denoising-steps 8


python custom/scripts/infer_groot_so101.py \
      --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251204_215410259/best \
      --task "grasp the red cube" \
      --action-horizon 4 \
      --log-file inference.log

python custom/scripts/infer_groot_so101.py \
      --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251204_215410259/best \
      --task "pick the red cube from the table" \
      --action-horizon 8 \
      --go-home-first \
      --log-file inference.log

python custom/scripts/infer_groot_so101.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251204_215410259/best \
    --task "pick the red cube from the table" \
    --action-horizon 12 \
    --denoising-steps 2 \
    --go-home-first \
    --log-file inference_fast.log

python custom/scripts/infer_groot_so101.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251204_215410259/best \
    --task "pick the red cube from the table" \
    --action-horizon 12 \
    --denoising-steps 2 \
    --go-home-first \
    --record-imgs \
    --log-file inference_debug.log