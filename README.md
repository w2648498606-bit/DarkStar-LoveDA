# LoveDA 七类遥感语义分割

本项目面向 LoveDA 城市/乡村遥感场景的七类语义分割，通过可控实验筛选模型和训练策略，并使用独立验证集与 Hidden Test 检查泛化表现。

建议按以下顺序查看仓库：

1. 当前 README：最终路线与主要结果
2. [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md)：独立验证、诊断实验与历史记录
3. [`train_e090.py`](train_e090.py)：最终实验配置及统一训练流程入口

## 1. Project Overview

项目主线：

**基础模型与架构筛选 → 多尺度训练与稀有类别采样 → 分类别及城乡分域诊断 → 泛化评估 → 最终轻量优化**

首页只展示最终主线和主要结果。未进入最终方案的探索、负结果及诊断实验保留在 [`docs/`](docs/) 和 [`logs/`](logs/) 中。

## 2. Final Model

- **Architecture：** MiT-B2 编码器 + UPerNet 风格解码器
- **Multi-scale training：** 使用多尺度增强适应不同尺度的地物
- **Rare-class sampling：** 提高少数类别出现在有效训练裁剪中的频率
- **Lightweight class-aware training：** 仅在训练阶段加入类别频率相关的轻量扰动，推理阶段仍使用原始模型输出，不增加额外推理分支

最终训练配置入口为 [`train_e090.py`](train_e090.py)，主体流程复用 [`train.py`](train.py) 中的统一训练框架。

## 3. Main Results

| Evaluation protocol | Experiment | mIoU | Purpose |
|---|---|---:|---|
| Independent validation | E090 | **0.545243** | 模型选择与独立验证；最佳轮次为 epoch 5 |
| Hidden Test | E090 单模型直接预测 | **0.525061**（项目记录） | 使用最佳检查点与原始模型输出，无 TTA、无集成 |

“Independent validation”仅指验证样本未参与对应训练的实验。使用 Train+Val 继续训练的运行不再具有独立验证意义，因此单独归入诊断性/非独立实验，不参与模型选择，也不作为首页主结果。完整口径见 [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md)。

第三方检查点复现与本项目训练结果分开记录。上表不包含第三方检查点的直接复现成绩；具体来源边界见 [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md#third-party-checkpoint--initialization-runs)。E090 也不应描述为“从零训练”：它沿用了本项目 E007 训练得到的编码器初始化。

## 4. Experimental Process

实验判断并非只看单个最高数字，主要依据包括：

- 独立验证集 mIoU
- 分类别 IoU 与弱类别表现
- Urban/Rural 分域表现
- Hidden Test 结果
- 控制变量实验中的成功、失败与负结果

历史记录包括 **74 个实验目录索引**和 **70 份 history 文件**。详细实验术语、配置与结果均保留在 [`logs/`](logs/) 及 [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md) 中。

## 5. Repository Structure

| Path | Description |
|---|---|
| [`train_e090.py`](train_e090.py) | E090 最终实验配置与训练入口 |
| [`train.py`](train.py) | 公共训练与验证流程 |
| [`predict_e090_lite_direct_test.py`](predict_e090_lite_direct_test.py) | E090 Hidden Test 单模型直接预测入口 |
| [`models/`](models/) | 模型结构实现 |
| [`datasets/`](datasets/) | LoveDA 数据读取与标签映射 |
| [`logs/`](logs/) | 历史训练记录与实验元数据 |
| [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md) | 按评测口径整理的实验索引 |

## 6. Data and Reproduction Notes

本仓库不重新分发 LoveDA 原始影像，也不包含模型权重、预测掩膜或提交压缩包。请先按 LoveDA 的官方要求获取数据，并整理为以下结构：

```text
LoveDA/
├── Train/
│   ├── Urban/images_png/  and masks_png/
│   └── Rural/images_png/  and masks_png/
├── Val/
│   ├── Urban/images_png/  and masks_png/
│   └── Rural/images_png/  and masks_png/
└── Test/
    ├── Urban/images_png/
    └── Rural/images_png/
```

训练入口不再依赖固定的 AutoDL 项目目录；在仓库根目录执行，并显式传入数据路径：

```bash
python train_e090.py --data-root /path/to/LoveDA
```

上述命令解决的是项目位置和数据路径的可移植性。若要严格复现 E090，还需将 E007 编码器初始化权重放到以下相对路径；该权重未随仓库上传：

```text
outputs/experiments/E007_segformer_b2_20ep/checkpoints/best_miou.pth
```

因此，本仓库提供可审计的源码、配置和实验记录，但不能在缺少数据与初始化权重时直接完成同口径复现。环境与标签约定见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

## 7. Public Release Scope

本仓库是项目完成后整理的公开展示与归档版本。发布时精简了重复实验入口和大体积生成物，保留最终源码、实验日志和结构化索引。因此，公开提交历史反映的是公开版本的整理过程，不等同于完整实验开发时间线。
