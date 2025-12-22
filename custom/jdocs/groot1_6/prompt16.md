# 1. Collect traces
python custom/scripts/ver1_6/infer_groot_so101_trace.py \
    --checkpoint outputs/groot_1_6_augmented_*/checkpoint-45000 \
    --task "pick the red cube and place it on the plate" \
    --duration 30

# 2. Analyze traces
python custom/scripts/ver1_6/analyze_inference_trace.py \
    --trace-dir outputs/inference_traces/trace_20251221_215113 \
    --show-plots