#!/bin/bash

# LightUNet 全监督 5-Seed 批量实验脚本
# 运行所有数据集的全监督实验

set -e

# ============ 配置参数 ============
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

# 所有数据集
DATASETS=("nuaa_sirst" "irstd1k" "sirst3" "nudt_sirst")

# 5个随机种子
SEEDS=(42 123 456 789 1024)

# ============ 辅助函数 ============
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

run_single_experiment() {
    local dataset=$1
    local seed=$2

    echo ">>> 开始: ${dataset} - Seed ${seed}"

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
        echo "✅ ${dataset} - Seed ${seed} 完成"
        return 0
    else
        echo "❌ ${dataset} - Seed ${seed} 失败"
        return 1
    fi
}

# ============ 主程序 ============
print_header "LightUNet 全监督批量实验"
echo "数据集: ${DATASETS[@]}"
echo "种子: ${SEEDS[@]}"
echo "总实验数: $((${#DATASETS[@]} * ${#SEEDS[@]}))"
echo ""

# 统计
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SEEDS[@]}))
CURRENT_EXPERIMENT=0
FAILED_EXPERIMENTS=()

# 遍历所有数据集和种子
for dataset in "${DATASETS[@]}"; do
    print_header "数据集: ${dataset}"

    for seed in "${SEEDS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        echo ""
        echo "[${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] 运行实验..."

        if ! run_single_experiment "${dataset}" ${seed}; then
            FAILED_EXPERIMENTS+=("${dataset}_seed${seed}")
        fi
    done
done

# ============ 总结 ============
print_header "实验总结"
echo "总实验数: ${TOTAL_EXPERIMENTS}"
echo "成功: $((TOTAL_EXPERIMENTS - ${#FAILED_EXPERIMENTS[@]}))"
echo "失败: ${#FAILED_EXPERIMENTS[@]}"

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo ""
    echo "🎉 所有实验成功完成!"
else
    echo ""
    echo "⚠️  失败的实验:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
fi

echo ""
echo "结果目录结构:"
for dataset in "${DATASETS[@]}"; do
    echo "  ${dataset}/"
    for seed in "${SEEDS[@]}"; do
        echo "    ├── focal_5e-4_cosine_300_seed${seed}/"
    done
done
