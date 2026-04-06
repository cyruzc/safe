# Trizone

弱监督红外小目标检测实验代码。

这个仓库主要做三类对比：
- `full`: 全监督
- `point`: 普通点监督
- `safe`: 点监督 + SAFE 先验约束

训练阶段每个 epoch 只验证 `IoU`，用 `best_iou.pt` 选最优模型；训练结束后会自动对最优模型跑一次完整指标，并保存到 `final_metrics.json`。

## 1. 环境
创建环境，激活环境（这里例如叫safe）
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate safe
```

## 2. 数据目录

当前内置数据集：
- `sirst3`
- `irstd1k`
- `nuaa_sirst`
- `nudt_sirst`

以 `SIRST3` 为例，目录结构应为：

```text
datasets/SIRST3/
├── images/
├── masks/
├── masks_centroid/
├── masks_coarse/
└── img_idx/
    ├── train_SIRST3.txt
    └── test_SIRST3.txt
```

## 3. 实验目录规则

自动生成的实验目录：

```text
experiments/runs/{dataset}/{model}/{supervision}/{exp_tag}_seed{seed}
experiments/priors/{dataset}/{label_mode}/{prior_tag}
```

其中：
- `supervision`: `full`, `point_centroid`, `point_coarse`, `safe_centroid`, `safe_coarse`
- `exp_tag`: 例如 `base`, `cosine`, `dog_wslcm_iw002_ow02_warm10`

训练输出目录里会有：
- `config.json`
- `history.csv`
- `last.pt`
- `best_iou.pt`
- `final_metrics.json`

## 4. 先生成 SAFE priors

如果跑 `safe`，先生成 priors。

例子：`SIRST3 + centroid + dog/wslcm`

```bash
python generate_priors.py \
  --dataset-name sirst3 \
  --label-mode centroid \
  --inner-method dog \
  --outer-method wslcm
```

默认会生成到：

```text
experiments/priors/sirst3/centroid/dog_wslcm_p999_p995
```

其中真正训练要用的是：

```text
experiments/priors/sirst3/centroid/dog_wslcm_p999_p995/priors/inner_dog_p99_9
experiments/priors/sirst3/centroid/dog_wslcm_p999_p995/priors/outer_wslcm_p99_5
```

## 5. 训练示例

### 5.1 全监督

```bash
python train.py \
  --dataset-name sirst3 \
  --model-name acm \
  --method full \
  --exp-tag base \
  --seed 42 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-3 \
  --num-workers 6 \
  --device cuda \
  --amp
```

### 5.2 普通点监督

`centroid` 版本：

```bash
python train.py \
  --dataset-name sirst3 \
  --label-mode centroid \
  --model-name acm \
  --method point \
  --exp-tag base \
  --seed 42 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-3 \
  --num-workers 6 \
  --device cuda \
  --amp
```

### 5.3 SAFE 点监督

```bash
python train.py \
  --dataset-name sirst3 \
  --label-mode centroid \
  --model-name acm \
  --method safe \
  --exp-tag dog_wslcm_iw002_ow02_warm10 \
  --seed 42 \
  --inner-prior-dir experiments/priors/sirst3/centroid/dog_wslcm_p999_p995/priors/inner_dog_p99_9 \
  --outer-prior-dir experiments/priors/sirst3/centroid/dog_wslcm_p999_p995/priors/outer_wslcm_p99_5 \
  --inner-loss-weight 0.02 \
  --outer-loss-weight 0.2 \
  --prior-warmup-epochs 10 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-3 \
  --num-workers 6 \
  --device cuda \
  --amp
```

这个命令会自动保存到：

```text
experiments/runs/sirst3/acm/safe_centroid/dog_wslcm_iw002_ow02_warm10_seed42
```

## 6. 查看结果

训练过程看：
- `history.csv`: 每个 epoch 的 loss 和 IoU

最终成绩看：
- `final_metrics.json`: 最优模型的完整指标

最优模型文件：
- `best_iou.pt`

## 7. 单独评估已有 checkpoint

```bash
python evaluate.py \
  --dataset-name sirst3 \
  --checkpoint experiments/runs/sirst3/acm/safe_centroid/dog_wslcm_iw002_ow02_warm10_seed42/best_iou.pt \
  --device cuda \
  --amp
```

## 8. 推荐实验矩阵

建议至少跑这三组：
- `full`
- `point_centroid` / `point_coarse`
- `safe_centroid` / `safe_coarse`

例如对同一个模型 `acm`：
- `sirst3/acm/full/base_seed42`
- `sirst3/acm/point_centroid/base_seed42`
- `sirst3/acm/point_coarse/base_seed42`
- `sirst3/acm/safe_centroid/dog_wslcm_iw002_ow02_warm10_seed42`
- `sirst3/acm/safe_coarse/dog_wslcm_iw002_ow02_warm10_seed42`

最后统一比较各目录下的 `final_metrics.json`。
