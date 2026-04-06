# SAFE

弱监督红外小目标检测实验代码。

这个仓库主要做三类对比：
- `full`: 全监督
- `point`: 普通点监督
- `safe`: SAFE方法 - 点监督 + 三区域先验约束

训练阶段每个 epoch 只验证 `IoU`，用 `best_iou.pt` 选最优模型；训练结束后会自动对最优模型跑一次完整指标，并保存到 `final_metrics.json`。

## 📁 项目结构

```
.
├── scripts/                    # 实验和分析脚本
│   ├── test_1epoch.sh         # 1-epoch快速测试
│   ├── run_unet_experiments.sh # 完整实验运行脚本
│   ├── start_experiments.sh    # 交互式启动器
│   ├── monitor_experiments.sh  # 实时监控进度
│   └── analyze_unet_experiments.py # 结果分析
├── models/                     # 模型定义
│   ├── model_lightweight_unet.py
│   └── model_ACM.py
├── data.py                     # 数据加载
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── generate_priors.py          # 先验生成
├── losses.py                   # 损失函数
└── metrics.py                  # 评估指标
```

## 🚀 快速开始

### 1. 环境配置
创建环境，激活环境（这里例如叫safe）
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate safe
```

### 2. 数据目录

当前内置数据集：
- `irstd1k`
- `nuaa_sirst`
- `nudt_sirst`
- `sirst3`

以 `nuaa_sirst` 为例，目录结构应为：

```text
datasets/NUAA-SIRST/
├── images/
├── masks/
├── masks_centroid/
├── masks_coarse/
└── img_idx/
    ├── train_NUAA-SIRST.txt
    └── test_NUAA-SIRST.txt
```

### 3. 运行快速测试

验证实验流程是否正常：

```bash
cd scripts
./test_1epoch.sh
```

这会运行5组实验（全监督、点监督、SAFE方法）的1-epoch版本，确保一切正常。

### 4. 生成SAFE先验

如果跑 `safe` 方法，需要先生成三区域先验。

例子：`nuaa_sirst + centroid + dog/wslcm`

```bash
python generate_priors.py \
  --dataset-name nuaa_sirst \
  --label-mode centroid \
  --inner-method dog \
  --outer-method wslcm
```

默认会生成到：

```text
experiments/priors/nuaa_sirst/centroid/dog_wslcm_p999_p995
```

## 🧪 实验运行

### 交互式启动

```bash
cd scripts
./start_experiments.sh
```

选择实验模式：
- 串行运行（推荐，稳定）
- 仅监控现有实验
- 查看使用说明

### 实时监控

在另一个终端监控进度：

```bash
cd scripts
./monitor_experiments.sh
```

### 查看结果

```bash
# 快速查看结果概览
./experiment_logs/view_results.sh

# 详细分析结果
python analyze_unet_experiments.py
```

## 📊 实验目录规则

自动生成的实验目录：

```text
experiments/runs/{dataset}/{model}/{supervision}/{exp_tag}_seed{seed}
experiments/priors/{dataset}/{label_mode}/{prior_tag}
```

其中：
- `supervision`: `full`, `point_centroid`, `point_coarse`, `safe_centroid`, `safe_coarse`
- `exp_tag`: 例如 `base`, `focal_5e-4_cosine_300`

训练输出目录里会有：
- `config.json`
- `history.csv`
- `last.pt`
- `best_iou.pt`
- `final_metrics.json`

## 📈 查看结果

训练过程看：
- `history.csv`: 每个 epoch 的 loss 和 IoU

最终成绩看：
- `final_metrics.json`: 最优模型的完整指标

最优模型文件：
- `best_iou.pt`

## 📖 详细使用说明

完整的使用指南请参考：
- `EXPERIMENT_GUIDE.md` - 实验设计详细说明
- `UNET_EXPERIMENTS.md` - LightweightUNet实验指南

## 🔧 单独评估已有 checkpoint

```bash
python evaluate.py \
  --dataset-name nuaa_sirst \
  --checkpoint experiments/runs/nuaa_sirst/lightweight_unet/safe_centroid/focal_5e-4_cosine_300_seed42/best_iou.pt \
  --device cuda \
  --amp
```

## 🎯 推荐实验矩阵

建议至少跑这三组对比：
- `full` - 全监督上界
- `point_centroid` / `point_coarse` - 传统点监督下界
- `safe_centroid` / `safe_coarse` - SAFE方法 (点监督+三区域先验)

例如对同一个模型 `lightweight_unet`：
- `nuaa_sirst/lightweight_unet/full/focal_5e-4_cosine_300_seed42`
- `nuaa_sirst/lightweight_unet/point_centroid/focal_5e-4_cosine_300_seed42`
- `nuaa_sirst/lightweight_unet/point_coarse/focal_5e-4_cosine_300_seed42`
- `nuaa_sirst/lightweight_unet/safe_centroid/focal_5e-4_cosine_300_seed42`
- `nuaa_sirst/lightweight_unet/safe_coarse/focal_5e-4_cosine_300_seed42`

最后统一比较各目录下的 `final_metrics.json`。
