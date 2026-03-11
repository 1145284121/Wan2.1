#!/bin/bash
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys

for exp in 1_baseline 2_vectorized 3_compile_default 4_compile_fullgraph 5_triton; do
    echo "=== Running $exp ==="
    $NSYS profile -t cuda,nvtx --force-overwrite true \
        -o "profile_${exp}" \
        python3 test_attention_block_profile.py "$exp" 2>&1 | tail -5
    echo ""
done
echo "Done. Files:"
ls -lh profile_*.nsys-rep
