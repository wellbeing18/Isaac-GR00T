python custom/scripts/ver1_6/infer_groot_so101_trace.py \
    --checkpoint outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 30

# 2. Analyze traces
python custom/scripts/ver1_6/analyze_inference_trace.py \
    --trace-dir outputs/inference_traces/trace_20251221_181859 \
    --show-plots