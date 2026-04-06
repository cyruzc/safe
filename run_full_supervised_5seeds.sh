#!/bin/bash

# LightUNet 5-Seed Full Supervision Experiment Runner
# 全监督 5 seed 实验运行脚本

set -e

# ============ 配置参数 ============
DATASET_NAME="${1:-nuaa_sirst}"  # 默认数据集: nuaa_sirst
MODEL_NAME="lightweight_unet"
METHOD="full"
BATCH_SIZE=16
EPOCHS=300
LR=5e-4
WEIGHT_DECAY=1e-4
SCHEDULER="cosine"
LOSS_TYPE="focal"
NUM_WORKERS=6
DEVICE="cuda"
AMP=true

# 5个随机种子
SEEDS=(42 123 456 789 1024)

# ============ 辅助函数 ============
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_section() {
    echo ""
    echo ">>> $1"
}

run_experiment() {
    local dataset=$1
    local seed=$2

    print_header "开始实验: ${dataset} - Seed ${seed}"

    python train.py \
        --dataset-name "${dataset}" \
        --model-name "${MODEL_NAME}" \
        --method "${METHOD}" \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --scheduler ${SCHEDULER} \
        --loss-type ${LOSS_TYPE} \
        --seed ${seed} \
        --num-workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --amp \
        --exp-tag focal_5e-4_cosine_300

    if [ $? -eq 0 ]; then
        print_section "✅ 实验 ${dataset} - Seed ${seed} 完成!"
    else
        print_section "❌ 实验 ${dataset} - Seed ${seed} 失败!"
        return 1
    fi
}

# ============ 主程序 ============
print_header "LightUNet 全监督 5-Seed 实验"
echo "数据集: ${DATASET_NAME}"
echo "模型: ${MODEL_NAME}"
echo "方法: ${METHOD}"
echo "种子: ${SEEDS[@]}"
echo ""

# 检查CUDA可用性
if [ "${DEVICE}" == "cuda" ]; then
    print_section "检查 CUDA 环境"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
fi

# 运行5个seed的实验
FAILED_SEEDS=()

for seed in "${SEEDS[@]}"; do
    if ! run_experiment "${DATASET_NAME}" ${seed}; then
        FAILED_SEEDS+=(${seed})
    fi
done

# ============ 总结 ============
print_header "实验总结"

if [ ${#FAILED_SEEDS[@]} -eq 0 ]; then
    echo "🎉 所有实验成功完成!"
    echo ""
    echo "结果目录:"
    for seed in "${SEEDS[@]}"; do
        echo "  experiments/runs/${DATASET_NAME}/${MODEL_NAME}/${METHOD}/focal_5e-4_cosine_300_seed${seed}/"
    done
else
    echo "⚠️  部分实验失败!"
    echo "失败的种子: ${FAILED_SEEDS[@]}"
    echo ""
    echo "请检查日志并重新运行失败的实验。"
    exit 1
fi

print_section "查看结果"
echo "训练过程: experiments/runs/${DATASET_NAME}/${MODEL_NAME}/${METHOD}/focal_5e-4_cosine_300_seed<seed>/history.csv"
echo "最终指标: experiments/runs/${DATASET_NAME}/${MODEL_NAME}/${METHOD}/focal_5e-4_cosine_300_seed<seed>/final_metrics.json"
echo ""
echo "示例查看:"
echo "  cat experiments/runs/${DATASET_NAME}/${MODEL_NAME}/${METHOD}/focal_5e-4_cosine_300_seed42/final_metrics.json"
