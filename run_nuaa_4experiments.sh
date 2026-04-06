#!/bin/bash

# NUAA-SIRST 数据集 4 个核心实验（无全监督）
# 验证真正的单点监督和 SAFE prior 的效果

set -e

echo "========================================"
echo "  NUAA-SIRST 4 个核心实验"
echo "========================================"
echo ""
echo "实验列表："
echo "  1. Point + Coarse（单点监督）"
echo "  2. Point + Centroid（单点监督）"
echo "  3. SAFE + Coarse (w=1.0)"
echo "  4. SAFE + Centroid (w=1.0)"
echo ""
echo "开始时间：$(date)"
echo ""

# 创建日志目录
mkdir -p logs/nuaa_4exp

# 实验 1: Point + Coarse
echo "========================================"
echo "实验 1/4: Point + Coarse"
echo "========================================"
python train.py \
  --dataset-name nuaa_sirst \
  --method point \
  --label-mode coarse \
  --model-name lightweight_unet \
  --epochs 300 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --loss-type focal \
  --scheduler cosine \
  --exp-tag point_coarse_300 \
  --seed 42 \
  2>&1 | tee logs/nuaa_4exp/01_point_coarse.log

echo ""
echo "✅ 实验 1 完成！"
echo ""

# 实验 2: Point + Centroid
echo "========================================"
echo "实验 2/4: Point + Centroid"
echo "========================================"
python train.py \
  --dataset-name nuaa_sirst \
  --method point \
  --label-mode centroid \
  --model-name lightweight_unet \
  --epochs 300 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --loss-type focal \
  --scheduler cosine \
  --exp-tag point_centroid_300 \
  --seed 42 \
  2>&1 | tee logs/nuaa_4exp/02_point_centroid.log

echo ""
echo "✅ 实验 2 完成！"
echo ""

# 实验 3: SAFE + Coarse (w=1.0)
echo "========================================"
echo "实验 3/4: SAFE + Coarse (prior weight=1.0)"
echo "========================================"
python train.py \
  --dataset-name nuaa_sirst \
  --method safe \
  --label-mode coarse \
  --model-name lightweight_unet \
  --epochs 300 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --loss-type focal \
  --scheduler cosine \
  --inner-loss-weight 0.02 \
  --outer-loss-weight 0.2 \
  --prior-warmup-epochs 10 \
  --exp-tag safe_coarse_inner002_outer02_warmup10 \
  --seed 42 \
  2>&1 | tee logs/nuaa_4exp/03_safe_coarse.log

echo ""
echo "✅ 实验 3 完成！"
echo ""

# 实验 4: SAFE + Centroid (w=1.0)
echo "========================================"
echo "实验 4/4: SAFE + Centroid (prior weight=1.0)"
echo "========================================"
python train.py \
  --dataset-name nuaa_sirst \
  --method safe \
  --label-mode centroid \
  --model-name lightweight_unet \
  --epochs 300 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --loss-type focal \
  --scheduler cosine \
  --inner-loss-weight 0.02 \
  --outer-loss-weight 0.2 \
  --prior-warmup-epochs 10 \
  --exp-tag safe_centroid_inner002_outer02_warmup10 \
  --seed 42 \
  2>&1 | tee logs/nuaa_4exp/04_safe_centroid.log

echo ""
echo "✅ 实验 4 完成！"
echo ""

echo "========================================"
echo "  所有 4 个实验完成！"
echo "========================================"
echo ""
echo "完成时间：$(date)"
echo ""
echo "实验结果汇总："
bash extract_nuaa_4exp.sh
